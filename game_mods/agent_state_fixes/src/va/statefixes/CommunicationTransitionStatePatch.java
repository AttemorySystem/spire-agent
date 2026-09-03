package va.statefixes;

import com.evacipated.cardcrawl.modthespire.lib.SpirePatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePostfixPatch;
import com.megacrit.cardcrawl.actions.GameActionManager;
import com.megacrit.cardcrawl.cards.SoulGroup;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.events.AbstractEvent;
import com.megacrit.cardcrawl.neow.NeowEvent;
import com.megacrit.cardcrawl.neow.NeowReward;
import com.megacrit.cardcrawl.rooms.AbstractRoom;
import com.megacrit.cardcrawl.screens.select.GridCardSelectScreen;
import com.megacrit.cardcrawl.vfx.AbstractGameEffect;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Expose whether a command still has queued game actions or visual effects.
 *
 * <p>CommunicationMod may report a new choice screen before effects such as
 * ShowCardAndObtainEffect have committed their permanent state.  A fast
 * caller can otherwise leave the event before the selected card enters the
 * deck.  The harness uses this signal at its shared live/replay transition
 * boundary.</p>
 */
@SpirePatch(
    cls = "communicationmod.GameStateConverter",
    method = "getGameState",
    requiredModId = "CommunicationMod",
    optional = true
)
public final class CommunicationTransitionStatePatch {
    private CommunicationTransitionStatePatch() {}

    private static String neowGridOperation(AbstractEvent event) {
        if (!(event instanceof NeowEvent)) {
            return "";
        }
        try {
            Field rewardsField = NeowEvent.class.getDeclaredField("rewards");
            Field activatedField = NeowReward.class.getDeclaredField("activated");
            rewardsField.setAccessible(true);
            activatedField.setAccessible(true);
            for (Object value : (ArrayList<?>) rewardsField.get(event)) {
                NeowReward reward = (NeowReward) value;
                if (!activatedField.getBoolean(reward)) {
                    continue;
                }
                switch (reward.type) {
                    case REMOVE_CARD:
                    case REMOVE_TWO:
                        return "REMOVE";
                    case TRANSFORM_CARD:
                    case TRANSFORM_TWO_CARDS:
                        return "TRANSFORM";
                    case UPGRADE_CARD:
                        return "UPGRADE";
                    default:
                        return "";
                }
            }
        } catch (ReflectiveOperationException | SecurityException error) {
            return "";
        }
        return "";
    }

    @SuppressWarnings("unchecked")
    private static void exposeGridOperation(
        HashMap<String, Object> state,
        AbstractRoom room
    ) {
        if (!"GRID".equals(String.valueOf(state.get("screen_type")))) {
            return;
        }
        Object value = state.get("screen_state");
        if (!(value instanceof HashMap)) {
            return;
        }
        GridCardSelectScreen grid = AbstractDungeon.gridSelectScreen;
        String operation =
            grid.forPurge
                ? "REMOVE"
                : grid.forTransform
                    ? "TRANSFORM"
                    : grid.forUpgrade
                        ? "UPGRADE"
                        : neowGridOperation(room == null ? null : room.event);
        if (!operation.isEmpty()) {
            ((HashMap<String, Object>) value).put("grid_operation", operation);
        }
    }

    private static boolean blocksDecisionBoundary(AbstractGameEffect effect) {
        if (effect == null) {
            return false;
        }
        String name = effect.getClass().getSimpleName();
        switch (name) {
            case "ShowCardAndObtainEffect":
            case "FastCardObtainEffect":
            case "PurgeCardEffect":
            case "ObtainPotionEffect":
            case "ObtainKeyEffect":
            case "RainingGoldEffect":
                return true;
            default:
                return false;
        }
    }

    private static boolean eventAnimationPending() {
        if (AbstractDungeon.getCurrRoom() == null) {
            return false;
        }
        AbstractEvent event = AbstractDungeon.getCurrRoom().event;
        if (event == null) {
            return false;
        }
        String className = event.getClass().getName();
        String timerName;
        if (className.equals("com.megacrit.cardcrawl.events.city.TheJoust")) {
            timerName = "joustTimer";
        } else if (
            className.equals(
                "com.megacrit.cardcrawl.events.shrines.GremlinMatchGame"
            )
        ) {
            timerName = "waitTimer";
        } else {
            return false;
        }
        try {
            Field timer = event.getClass().getDeclaredField(timerName);
            timer.setAccessible(true);
            return timer.getFloat(event) > 0.0F;
        } catch (ReflectiveOperationException | SecurityException error) {
            return true;
        }
    }

    @SpirePostfixPatch
    public static HashMap<String, Object> exposePendingTransition(
        HashMap<String, Object> __result
    ) {
        GameActionManager manager = AbstractDungeon.actionManager;
        boolean actionsPending = manager != null && !manager.isEmpty();
        String actionPhase =
            manager == null || manager.phase == null
                ? "UNAVAILABLE"
                : manager.phase.toString();
        String currentAction =
            manager == null || manager.currentAction == null
                ? ""
                : manager.currentAction.getClass().getSimpleName();
        ArrayList<String> effectNames = new ArrayList<>();
        if (AbstractDungeon.effectList != null) {
            for (AbstractGameEffect effect : AbstractDungeon.effectList) {
                if (blocksDecisionBoundary(effect)) {
                    effectNames.add(effect.getClass().getSimpleName());
                }
            }
        }
        if (AbstractDungeon.effectsQueue != null) {
            for (AbstractGameEffect effect : AbstractDungeon.effectsQueue) {
                if (blocksDecisionBoundary(effect)) {
                    effectNames.add(effect.getClass().getSimpleName());
                }
            }
        }
        boolean animationPending = eventAnimationPending();
        if (animationPending) {
            effectNames.add("EventAnimation");
        }
        AbstractRoom room = AbstractDungeon.getCurrRoom();
        exposeGridOperation(__result, room);
        if (room != null && SoulGroup.isActive()) {
            effectNames.add("CardMovement");
        }
        if (
            AbstractDungeon.player != null &&
            AbstractDungeon.player.isEscaping
        ) {
            effectNames.add("PlayerEscape");
        }
        if (
            room != null && room.isBattleOver &&
            room.phase == AbstractRoom.RoomPhase.COMBAT
        ) {
            effectNames.add("BattleEnding");
        }

        __result.put(
            "transition_pending",
            actionsPending || !effectNames.isEmpty()
        );
        // CommunicationMod omits current_action when it is null.  Emit both
        // queue signals unconditionally so a missing field means a broken
        // state contract, while an empty string explicitly means idle.
        __result.put("action_phase", actionPhase);
        __result.put("current_action", currentAction);
        __result.put("pending_effect_count", effectNames.size());
        __result.put("pending_effects", effectNames);
        return __result;
    }
}
