package va.statefixes;

import com.evacipated.cardcrawl.modthespire.lib.SpirePatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePostfixPatch;
import com.megacrit.cardcrawl.map.MapRoomNode;
import java.util.HashMap;

/**
 * Preserve the base game's burning-elite marker in CommunicationMod's map
 * node JSON. CommunicationMod otherwise serializes burning and ordinary
 * elites identically as {@code symbol: "E"}.
 */
@SpirePatch(
    cls = "communicationmod.GameStateConverter",
    method = "convertMapRoomNodeToJson",
    paramtypez = {MapRoomNode.class},
    requiredModId = "CommunicationMod",
    optional = true
)
public final class CommunicationMapNodePatch {
    private CommunicationMapNodePatch() {}

    @SpirePostfixPatch
    public static HashMap<String, Object> exposeBurningElite(
        HashMap<String, Object> __result,
        MapRoomNode node
    ) {
        if (node.hasEmeraldKey) {
            __result.put("is_burning", true);
        }
        return __result;
    }
}
