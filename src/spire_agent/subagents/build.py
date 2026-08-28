"""Mini BuildAgent with an injected card picker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
import json
from typing import Any

from spire_agent.contracts import AgentKind, Decision, DecisionRequest
from spire_agent.tools.build_flow import (
    BuildError,
    continue_build,
    fast_decision,
    llm_decision,
    policy_decision,
)
from spire_agent.tools.run_keys import rest_policy
from .build_prompt import build_prompt
from spire_agent.subagents.llm import LLMMessage, LLMRequest, LLMResponse, PromptLanguage

from .agents import BuildAgent


class CardPicker(ABC):
    """Replaceable card-picking policy used by BuildAgent."""

    @abstractmethod
    def review(self, request: DecisionRequest) -> dict[str, Any]: ...

    @abstractmethod
    def approve(
        self, result: Mapping[str, Any], proposal: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    @abstractmethod
    def prompt_context(self, result: Mapping[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def decision_payload(
        self,
        result: Mapping[str, Any],
        *,
        command: str,
        approval: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def review_shop(self, request: DecisionRequest) -> dict[str, Any]: ...

    @abstractmethod
    def approve_shop(
        self, result: Mapping[str, Any], proposal: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    @abstractmethod
    def shop_prompt_context(self, result: Mapping[str, Any]) -> dict[str, Any]: ...

    def shop_decision_payload(
        self, result: Mapping[str, Any], approval: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return optional confirmed run-level data; existing pickers need none."""

        return {}


class BuildStage:
    def __init__(
        self,
        llm: object,
        card_picker: CardPicker,
        language: PromptLanguage | str,
        choice_policy: Callable[[DecisionRequest], Mapping[str, Any] | None],
    ) -> None:
        self._llm = llm
        self._card_picker = card_picker
        self._language = PromptLanguage.parse(language)
        self._choice_policy = choice_policy

    def try_decide(self, request: DecisionRequest) -> Decision | None:
        continued = continue_build(request)
        if continued is not None:
            return continued
        if request.scope.owner is not AgentKind.BUILD:
            return None
        decision = fast_decision(request)
        if decision is not None:
            return decision
        complete = getattr(self._llm, "complete", None)
        if not callable(complete):
            raise TypeError("BuildAgent LLM has no complete() method")
        if request.state.screen.type == "CARD_REWARD":
            return self._card_reward(request, complete)
        if request.state.screen.type == "SHOP_SCREEN":
            policy = self._card_picker.review_shop(request)
            prompt = build_prompt(
                request,
                self._language,
                policy_context=self._card_picker.shop_prompt_context(policy),
            )
            response = complete(prompt)
            raw_data = getattr(response, "data", None)
            if not isinstance(raw_data, Mapping):
                raise BuildError("LLM response data must be an object")
            try:
                return self._shop_decision(
                    request, response, prompt, policy, raw_data
                )
            except BuildError as error:
                repair = _repair_request(prompt, response, error)
                corrected = complete(repair)
                raw_data = getattr(corrected, "data", None)
                if not isinstance(raw_data, Mapping):
                    raise BuildError("LLM response data must be an object")
                return self._shop_decision(
                    request, corrected, repair, policy, raw_data
                )
        key_rule = self._choice_policy(request)
        legal_ids = (
            key_rule.get("legal_choice_ids")
            if isinstance(key_rule, Mapping)
            else None
        )
        prompt = build_prompt(
            request,
            self._language,
            choice_policy=key_rule,
        )
        response = complete(prompt)
        try:
            return llm_decision(
                request, response, prompt, legal_choice_ids=legal_ids
            )
        except BuildError as error:
            repair = _repair_request(prompt, response, error)
            return llm_decision(
                request, complete(repair), repair, legal_choice_ids=legal_ids
            )

    def _shop_decision(
        self,
        request: DecisionRequest,
        response: object,
        prompt: LLMRequest,
        policy: Mapping[str, Any],
        raw_data: Mapping[str, Any],
    ) -> Decision:
        approval = self._card_picker.approve_shop(policy, raw_data)
        effective = LLMResponse(
            approval["data"],
            raw_text=(
                str(getattr(response, "raw_text", ""))
                if approval["approved"]
                else json.dumps(
                    approval["data"], ensure_ascii=False, separators=(",", ":")
                )
            ),
            model=str(getattr(response, "model", "")),
            usage=getattr(response, "usage", {}),
        )
        proposed = llm_decision(request, effective, prompt)
        picker_payload = self._card_picker.shop_decision_payload(policy, approval)
        return Decision(
            proposed.command,
            proposed.source if approval["approved"] else "card_reward.shop_veto",
            proposed.reason,
            continuation=proposed.continuation,
            payload={
                **dict(proposed.payload),
                **picker_payload,
                "shop_card_policy": {
                    **dict(policy),
                    "llm_approved": approval["approved"],
                    "veto_reason": approval["veto_reason"],
                },
                "llm_proposal": dict(raw_data),
            },
            metrics=proposed.metrics,
        )

    def _card_reward(
        self,
        request: DecisionRequest,
        complete: Callable[[object], object],
    ) -> Decision:
        result = self._card_picker.review(request)
        direct = result.get("command")
        if isinstance(direct, str) and direct:
            return policy_decision(
                request,
                direct,
                "card_reward.policy",
                str(result["reason"]),
                payload=self._card_picker.decision_payload(result, command=direct),
            )

        prompt = build_prompt(
            request,
            self._language,
            policy_context=self._card_picker.prompt_context(result),
        )
        response = complete(prompt)
        raw_data = getattr(response, "data", None)
        if not isinstance(raw_data, Mapping):
            raise TypeError("BuildAgent LLM response data must be a mapping")
        approval = self._card_picker.approve(result, raw_data)
        effective = LLMResponse(
            approval["data"],
            raw_text=(
                str(getattr(response, "raw_text", ""))
                if approval["approved"]
                else json.dumps(
                    approval["data"], ensure_ascii=False, separators=(",", ":")
                )
            ),
            model=str(getattr(response, "model", "")),
            usage=getattr(response, "usage", {}),
        )
        proposed = llm_decision(
            request,
            effective,
            prompt,
            legal_choice_ids=result.get("allowed_choice_ids") or (),
        )
        return Decision(
            proposed.command,
            "card_reward.policy_llm",
            proposed.reason,
            continuation=proposed.continuation,
            payload={
                **dict(proposed.payload),
                **self._card_picker.decision_payload(
                    result,
                    command=proposed.command,
                    approval=approval,
                ),
                "llm_proposal": dict(raw_data),
            },
            metrics=proposed.metrics,
        )


def _repair_request(
    prompt: LLMRequest, response: object, error: BuildError
) -> LLMRequest:
    raw = str(getattr(response, "raw_text", "")) or json.dumps(
        dict(getattr(response, "data", {})), ensure_ascii=False
    )
    return LLMRequest(
        prompt.purpose,
        prompt.messages
        + (
            LLMMessage("assistant", raw),
            LLMMessage(
                "user",
                f"That response is invalid: {error}. Return one corrected JSON object.",
            ),
        ),
        prompt.response_schema,
    )


def create_build_agent(
    llm: object,
    card_picker: CardPicker,
    *,
    prompt_language: PromptLanguage | str = PromptLanguage.ENGLISH,
    choice_policy: Callable[
        [DecisionRequest], Mapping[str, Any] | None
    ] = rest_policy,
) -> BuildAgent:
    return BuildAgent(
        tool_stages=(BuildStage(llm, card_picker, prompt_language, choice_policy),)
    )


__all__ = ["BuildStage", "CardPicker", "create_build_agent"]
