package va.statefixes;

import com.evacipated.cardcrawl.modthespire.lib.SpirePatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePostfixPatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePrefixPatch;
import com.megacrit.cardcrawl.actions.GameActionManager;
import com.megacrit.cardcrawl.actions.common.BetterDiscardPileToHandAction;
import com.megacrit.cardcrawl.cards.AbstractCard;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.monsters.AbstractMonster;
import com.megacrit.cardcrawl.monsters.EnemyMoveInfo;
import com.megacrit.cardcrawl.monsters.beyond.TimeEater;
import com.megacrit.cardcrawl.orbs.AbstractOrb;
import com.megacrit.cardcrawl.powers.AbstractPower;
import com.megacrit.cardcrawl.powers.StasisPower;
import com.megacrit.cardcrawl.relics.AbstractRelic;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Expose per-turn card counters needed to resume a combat simulation after
 * every real-game action. Without them, each MCTS invocation starts as though
 * no card had been played this turn, breaking Pocketwatch, Art of War,
 * Normality, and Velvet Choker.
 */
@SpirePatch(
    cls = "communicationmod.GameStateConverter",
    method = "getCombatState",
    requiredModId = "CommunicationMod",
    optional = true
)
public final class CommunicationCombatStatePatch {
    private static final Field MONSTER_MOVE = privateField(
        AbstractMonster.class,
        "move"
    );
    private static final Field TIME_EATER_USED_HASTE = privateField(
        TimeEater.class,
        "usedHaste"
    );
    private static final Field RELIC_PULSE = privateField(
        AbstractRelic.class,
        "pulse"
    );
    private static final Field STASIS_CARD = privateField(
        StasisPower.class,
        "card"
    );
    private static final Field DISCARD_TO_HAND_SET_COST = privateField(
        BetterDiscardPileToHandAction.class,
        "setCost"
    );
    private static final Field DISCARD_TO_HAND_NEW_COST = privateField(
        BetterDiscardPileToHandAction.class,
        "newCost"
    );
    private static BetterDiscardPileToHandAction pendingDiscardToHandAction;
    private static final ArrayList<AbstractCard> pendingDiscardToHandCards =
        new ArrayList<>();
    private static int pendingDiscardToHandCost;

    private CommunicationCombatStatePatch() {}

    private static Field privateField(Class<?> owner, String name) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException(
                "Cannot expose required game state " + owner.getName() + "." + name,
                error
            );
        }
    }

    @SuppressWarnings("unchecked")
    private static void exposeDynamicCardState(
        HashMap<String, Object> serializedCard,
        AbstractCard card
    ) {
        serializedCard.put("misc", card.misc);
        if ("Gash".equals(card.cardID)) {
            serializedCard.put("base_damage", card.baseDamage);
        } else if ("Steam".equals(card.cardID)) {
            serializedCard.put("base_block", card.baseBlock);
        }
    }

    @SuppressWarnings("unchecked")
    private static void exposeCardState(
        HashMap<String, Object> combatState,
        String pileName,
        List<AbstractCard> cards
    ) {
        Object value = combatState.get(pileName);
        if (!(value instanceof List)) {
            return;
        }

        List<?> serializedCards = (List<?>) value;
        int count = Math.min(serializedCards.size(), cards.size());
        for (int i = 0; i < count; ++i) {
            Object serialized = serializedCards.get(i);
            if (serialized instanceof HashMap) {
                HashMap<String, Object> serializedCard =
                    (HashMap<String, Object>) serialized;
                AbstractCard card = cards.get(i);
                exposeDynamicCardState(serializedCard, card);
                // CommunicationMod's existing "cost" field is costForTurn.
                // Export AbstractCard.cost separately so the native simulator
                // can distinguish persistent Snecko randomization from
                // temporary discounts such as Mummified Hand.
                serializedCard.put("combat_cost", card.cost);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static void exposeDefectState(HashMap<String, Object> combatState) {
        Object playerValue = combatState.get("player");
        if (playerValue instanceof HashMap) {
            ((HashMap<String, Object>) playerValue).put(
                "orb_slots",
                AbstractDungeon.player.maxOrbs
            );
        }

        int lightning = 0;
        int frost = 0;
        GameActionManager actionManager = AbstractDungeon.actionManager;
        if (actionManager != null) {
            for (AbstractOrb orb : actionManager.orbsChanneledThisCombat) {
                if ("Lightning".equals(orb.ID)) {
                    ++lightning;
                } else if ("Frost".equals(orb.ID)) {
                    ++frost;
                }
            }
        }
        combatState.put("lightning_channeled_this_combat", lightning);
        combatState.put("frost_channeled_this_combat", frost);

        AbstractRelic emotionChip = AbstractDungeon.player.getRelic("Emotion Chip");
        boolean pending = false;
        if (emotionChip != null) {
            try {
                pending = RELIC_PULSE.getBoolean(emotionChip);
            } catch (IllegalAccessException error) {
                throw new IllegalStateException(
                    "Cannot read Emotion Chip pending state",
                    error
                );
            }
        }
        combatState.put("emotion_chip_pending", pending);
    }

    @SuppressWarnings("unchecked")
    private static void exposeMonsterState(HashMap<String, Object> combatState) {
        Object value = combatState.get("monsters");
        if (!(value instanceof List) || AbstractDungeon.getMonsters() == null) {
            return;
        }

        List<?> serializedMonsters = (List<?>) value;
        List<AbstractMonster> monsters = AbstractDungeon.getMonsters().monsters;
        int count = Math.min(serializedMonsters.size(), monsters.size());
        for (int i = 0; i < count; ++i) {
            AbstractMonster monster = monsters.get(i);
            Object serialized = serializedMonsters.get(i);
            if (!(serialized instanceof HashMap)) {
                continue;
            }
            HashMap<String, Object> serializedMonster =
                (HashMap<String, Object>) serialized;
            try {
                // CommunicationMod deliberately omits move_id while Runic
                // Dome hides the rendered intent.  The native simulator still
                // needs the already-rolled move to import the real combat
                // state.  Only expose it after the internal move object exists
                // so the harness can continue using move_id as its readiness
                // signal during combat entry.
                if (!serializedMonster.containsKey("move_id")) {
                    EnemyMoveInfo move =
                        (EnemyMoveInfo) MONSTER_MOVE.get(monster);
                    if (move != null) {
                        serializedMonster.put("move_id", move.nextMove);
                    }
                }

                if (monster instanceof TimeEater) {
                    serializedMonster.put(
                        "miscBool",
                        TIME_EATER_USED_HASTE.getBoolean(monster)
                    );
                }

                Object powersValue = serializedMonster.get("powers");
                if (powersValue instanceof List) {
                    List<?> serializedPowers = (List<?>) powersValue;
                    List<AbstractPower> powers = monster.powers;
                    int powerCount = Math.min(
                        serializedPowers.size(),
                        powers.size()
                    );
                    for (int powerIdx = 0; powerIdx < powerCount; ++powerIdx) {
                        AbstractPower power = powers.get(powerIdx);
                        Object serializedPower = serializedPowers.get(powerIdx);
                        if (!(power instanceof StasisPower)
                            || !(serializedPower instanceof HashMap)) {
                            continue;
                        }
                        Object serializedCard =
                            ((HashMap<String, Object>) serializedPower).get("card");
                        if (!(serializedCard instanceof HashMap)) {
                            continue;
                        }
                        exposeDynamicCardState(
                            (HashMap<String, Object>) serializedCard,
                            (AbstractCard) STASIS_CARD.get(power)
                        );
                    }
                }
            } catch (IllegalAccessException error) {
                throw new IllegalStateException(
                    "Cannot read required monster state",
                    error
                );
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static void exposePlayerFacing(HashMap<String, Object> combatState) {
        Object value = combatState.get("player");
        if (value instanceof HashMap) {
            ((HashMap<String, Object>) value).put(
                "facing_left",
                AbstractDungeon.player.flipHorizontal
            );
        }
    }

    /** Keep discard-to-hand temporary cost changes atomic across bridge frames. */
    @SpirePrefixPatch
    public static void preserveDiscardToHandCostOverride() {
        if (AbstractDungeon.player == null || AbstractDungeon.actionManager == null) {
            pendingDiscardToHandAction = null;
            pendingDiscardToHandCards.clear();
            return;
        }

        if (
            AbstractDungeon.actionManager.currentAction instanceof
                BetterDiscardPileToHandAction
        ) {
            BetterDiscardPileToHandAction action =
                (BetterDiscardPileToHandAction)
                    AbstractDungeon.actionManager.currentAction;
            if (action == pendingDiscardToHandAction) {
                return;
            }
            try {
                if (!DISCARD_TO_HAND_SET_COST.getBoolean(action)) {
                    return;
                }
                pendingDiscardToHandAction = action;
                pendingDiscardToHandCost = DISCARD_TO_HAND_NEW_COST.getInt(action);
                pendingDiscardToHandCards.clear();
                pendingDiscardToHandCards.addAll(
                    AbstractDungeon.player.discardPile.group
                );
            } catch (IllegalAccessException error) {
                throw new IllegalStateException(
                    "Cannot capture discard-to-hand cost override",
                    error
                );
            }
            return;
        }

        if (pendingDiscardToHandAction == null) {
            return;
        }
        for (AbstractCard card : pendingDiscardToHandCards) {
            if (AbstractDungeon.player.hand.contains(card)) {
                card.setCostForTurn(pendingDiscardToHandCost);
                card.applyPowers();
                pendingDiscardToHandAction = null;
                pendingDiscardToHandCards.clear();
                return;
            }
        }
    }

    @SpirePostfixPatch
    public static HashMap<String, Object> exposeTurnCardCounters(
        HashMap<String, Object> __result
    ) {
        if (AbstractDungeon.player == null) {
            return __result;
        }

        __result.put(
            "cards_played_this_turn",
            AbstractDungeon.player.cardsPlayedThisTurn
        );

        int attacks = 0;
        int skills = 0;
        GameActionManager actionManager = AbstractDungeon.actionManager;
        if (actionManager != null) {
            for (AbstractCard card : actionManager.cardsPlayedThisTurn) {
                if (card.type == AbstractCard.CardType.ATTACK) {
                    ++attacks;
                } else if (card.type == AbstractCard.CardType.SKILL) {
                    ++skills;
                }
            }
        }
        __result.put("attacks_played_this_turn", attacks);
        __result.put("skills_played_this_turn", skills);

        int powersPlayedThisCombat = 0;
        if (actionManager != null) {
            for (AbstractCard card : actionManager.cardsPlayedThisCombat) {
                if (card.type == AbstractCard.CardType.POWER) {
                    ++powersPlayedThisCombat;
                }
            }
        }
        __result.put("powers_played_this_combat", powersPlayedThisCombat);

        exposeCardState(
            __result,
            "draw_pile",
            AbstractDungeon.player.drawPile.group
        );
        exposeCardState(
            __result,
            "discard_pile",
            AbstractDungeon.player.discardPile.group
        );
        exposeCardState(
            __result,
            "hand",
            AbstractDungeon.player.hand.group
        );
        exposeCardState(
            __result,
            "exhaust_pile",
            AbstractDungeon.player.exhaustPile.group
        );
        exposeDefectState(__result);
        exposeMonsterState(__result);
        exposePlayerFacing(__result);

        AbstractRelic puzzle = AbstractDungeon.player.getRelic("Centennial Puzzle");
        __result.put(
            "centennial_puzzle_used_this_combat",
            puzzle != null && puzzle.grayscale
        );
        return __result;
    }
}
