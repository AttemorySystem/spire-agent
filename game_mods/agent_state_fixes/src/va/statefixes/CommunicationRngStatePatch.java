package va.statefixes;

import com.badlogic.gdx.math.RandomXS128;
import com.evacipated.cardcrawl.modthespire.lib.SpirePatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePostfixPatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePrefixPatch;
import com.evacipated.cardcrawl.modthespire.lib.SpireReturn;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.random.Random;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Expose and restore every seeded dungeon RNG at an agent decision boundary.
 *
 * <p>An action-only replay is not deterministic in Slay the Spire. Some game
 * actions advance an RNG while their animation is settling, so a different
 * frame schedule can leave the next action at a different random position.
 * The harness records these states before each live action and restores them
 * before executing the corresponding replay action.</p>
 */
public final class CommunicationRngStatePatch {
    public static final String STATE_KEY = "replay_rng_state";
    public static final String RESTORE_COMMAND = "rng_restore";

    private CommunicationRngStatePatch() {}

    private static LinkedHashMap<String, Random> dungeonRngs() {
        LinkedHashMap<String, Random> result = new LinkedHashMap<>();
        result.put("monster", AbstractDungeon.monsterRng);
        result.put("map", AbstractDungeon.mapRng);
        result.put("event", AbstractDungeon.eventRng);
        result.put("merchant", AbstractDungeon.merchantRng);
        result.put("card", AbstractDungeon.cardRng);
        result.put("treasure", AbstractDungeon.treasureRng);
        result.put("relic", AbstractDungeon.relicRng);
        result.put("potion", AbstractDungeon.potionRng);
        result.put("monster_hp", AbstractDungeon.monsterHpRng);
        result.put("ai", AbstractDungeon.aiRng);
        result.put("shuffle", AbstractDungeon.shuffleRng);
        result.put("card_random", AbstractDungeon.cardRandomRng);
        result.put("misc", AbstractDungeon.miscRng);
        return result;
    }

    private static ArrayList<Object> serialize(Random rng) {
        ArrayList<Object> value = new ArrayList<>();
        RandomXS128 random = rng.random;
        value.add(random.getState(0));
        value.add(random.getState(1));
        value.add(rng.counter);
        return value;
    }

    @SpirePatch(
        cls = "communicationmod.GameStateConverter",
        method = "getGameState",
        requiredModId = "CommunicationMod",
        optional = true
    )
    public static final class GameStatePatch {
        private GameStatePatch() {}

        @SpirePostfixPatch
        public static HashMap<String, Object> exposeRngState(
            HashMap<String, Object> __result
        ) {
            LinkedHashMap<String, Object> state = new LinkedHashMap<>();
            for (Map.Entry<String, Random> entry : dungeonRngs().entrySet()) {
                if (entry.getValue() != null) {
                    state.put(entry.getKey(), serialize(entry.getValue()));
                }
            }
            ArrayList<Object> cardBlizz = new ArrayList<>();
            cardBlizz.add(AbstractDungeon.cardBlizzRandomizer);
            cardBlizz.add(0L);
            cardBlizz.add(0);
            state.put("card_blizz", cardBlizz);
            __result.put(STATE_KEY, state);
            return __result;
        }
    }

    @SpirePatch(
        cls = "communicationmod.CommandExecutor",
        method = "executeCommand",
        paramtypez = {String.class},
        requiredModId = "CommunicationMod",
        optional = true
    )
    public static final class RestoreCommandPatch {
        private RestoreCommandPatch() {}

        @SpirePrefixPatch
        public static SpireReturn<Boolean> restore(String command) {
            String trimmed = command == null ? "" : command.trim();
            if (!trimmed.startsWith(RESTORE_COMMAND + " ")) {
                return SpireReturn.Continue();
            }

            HashMap<String, Random> rngs = dungeonRngs();
            String payload = trimmed.substring(RESTORE_COMMAND.length()).trim();
            for (String encoded : payload.split(";")) {
                String[] fields = encoded.split(",", -1);
                if (fields.length != 4) {
                    throw new IllegalArgumentException(
                        "invalid deterministic replay RNG entry: " + encoded
                    );
                }
                if (fields[0].equals("card_blizz")) {
                    AbstractDungeon.cardBlizzRandomizer =
                        Integer.parseInt(fields[1]);
                    continue;
                }
                if (!rngs.containsKey(fields[0])) {
                    throw new IllegalArgumentException(
                        "unknown deterministic replay RNG: " + fields[0]
                    );
                }
                Random rng = rngs.get(fields[0]);
                if (rng == null) {
                    throw new IllegalStateException(
                        "dungeon RNG is unavailable: " + fields[0]
                    );
                }
                long state0 = Long.parseLong(fields[1]);
                long state1 = Long.parseLong(fields[2]);
                int counter = Integer.parseInt(fields[3]);
                rng.random.setState(state0, state1);
                rng.counter = counter;
            }

            // The command changes no visible game object, so explicitly ask
            // CommunicationMod to publish a fresh state response.
            try {
                Class<?> listener = Class.forName(
                    "communicationmod.GameStateListener"
                );
                Method changed = listener.getMethod("registerStateChange");
                changed.invoke(null);
            } catch (ReflectiveOperationException error) {
                throw new IllegalStateException(
                    "cannot publish restored deterministic replay state",
                    error
                );
            }
            return SpireReturn.Return(true);
        }
    }
}
