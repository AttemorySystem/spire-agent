package va.statefixes;

import com.evacipated.cardcrawl.modthespire.lib.SpirePatch;
import com.evacipated.cardcrawl.modthespire.lib.SpirePostfixPatch;
import com.megacrit.cardcrawl.characters.AbstractPlayer;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.monsters.AbstractMonster;
import java.util.List;

/**
 * CommunicationMod queues targeted cards directly, bypassing the base game's
 * player-facing update in AbstractPlayer.playCard().  That changes Surrounded
 * damage in the Shield and Spear combat.  Apply the same update after a valid
 * CommunicationMod play command has been accepted.
 */
@SpirePatch(
    cls = "communicationmod.CommandExecutor",
    method = "executePlayCommand",
    paramtypez = {String[].class},
    requiredModId = "CommunicationMod",
    optional = true
)
public final class CommunicationPlayFacingPatch {
    private CommunicationPlayFacingPatch() {}

    @SpirePostfixPatch
    public static void faceTarget(String[] command) {
        AbstractPlayer player = AbstractDungeon.player;
        if (player == null || !player.hasPower("Surrounded")
            || command == null || command.length < 3
            || AbstractDungeon.getMonsters() == null) {
            return;
        }

        final int targetIndex;
        try {
            targetIndex = Integer.parseInt(command[2]);
        } catch (NumberFormatException ignored) {
            return;
        }

        List<AbstractMonster> monsters = AbstractDungeon.getMonsters().monsters;
        if (targetIndex < 0 || targetIndex >= monsters.size()) {
            return;
        }
        AbstractMonster target = monsters.get(targetIndex);
        player.flipHorizontal = target.drawX < player.drawX;
    }
}
