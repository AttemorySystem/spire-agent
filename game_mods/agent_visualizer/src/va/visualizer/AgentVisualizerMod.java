package va.visualizer;

import basemod.BaseMod;
import basemod.interfaces.PostInitializeSubscriber;
import basemod.interfaces.PostRenderSubscriber;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.GlyphLayout;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator;
import com.badlogic.gdx.utils.Align;
import com.badlogic.gdx.utils.JsonReader;
import com.badlogic.gdx.utils.JsonValue;
import com.evacipated.cardcrawl.modthespire.lib.SpireInitializer;
import com.google.gson.FieldNamingPolicy;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.megacrit.cardcrawl.core.Settings;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@SpireInitializer
public final class AgentVisualizerMod
    implements PostInitializeSubscriber, PostRenderSubscriber {

    private static final Gson GSON = new GsonBuilder()
        .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
        .create();
    private static final int MAX_SNAPSHOT_BYTES = 128 * 1024;
    private static final int BUILD_COLUMNS = 5;
    private static final Color PANEL_BACKGROUND =
        new Color(0.055f, 0.105f, 0.145f, 0.43f);
    private static final Color PANEL_SHADOW =
        new Color(0.01f, 0.025f, 0.05f, 0.18f);
    private static final Color PANEL_HEADER_WASH =
        new Color(0.12f, 0.42f, 0.52f, 0.11f);
    private static final Color PANEL_BORDER =
        new Color(0.24f, 0.86f, 0.96f, 0.38f);
    private static final Color PANEL_BORDER_SOFT =
        new Color(0.20f, 0.56f, 0.66f, 0.22f);
    private static final Color CHIP_BACKGROUND =
        new Color(0.075f, 0.145f, 0.19f, 0.58f);
    private static final Color CHIP_BORDER =
        new Color(0.25f, 0.75f, 0.84f, 0.26f);
    private static final Color ROUTE_CURRENT_BACKGROUND =
        new Color(0.18f, 0.82f, 0.94f, 0.20f);
    private static final Color ACCENT = new Color(0.28f, 0.91f, 1.0f, 1.0f);
    private static final Color ACCENT_GLOW =
        new Color(0.24f, 0.88f, 1.0f, 0.16f);
    private static final Color ACCENT_SECONDARY =
        new Color(0.92f, 0.36f, 0.92f, 0.72f);
    private static final Color TITLE = new Color(0.91f, 0.99f, 1.0f, 1.0f);
    private static final Color TEXT = new Color(0.86f, 0.94f, 0.98f, 1.0f);
    private static final Color MUTED = new Color(0.54f, 0.68f, 0.76f, 1.0f);
    private static final Color GOOD = new Color(0.48f, 0.95f, 0.62f, 1.0f);
    private static final Color WARN = new Color(1.0f, 0.77f, 0.28f, 1.0f);
    private static final Color BAD = new Color(1.0f, 0.40f, 0.42f, 1.0f);
    private final Path snapshotPath;
    private volatile OverlayState state;
    private BitmapFont titleFont;
    private BitmapFont bodyFont;
    private Texture pixel;
    private final GlyphLayout glyphLayout = new GlyphLayout();
    private FreeTypeFontGenerator titleFontGenerator;
    private FreeTypeFontGenerator bodyFontGenerator;
    private final Map<String, String> chineseCardNames = new HashMap<>();
    private final Map<String, String> chineseMonsterNames = new HashMap<>();
    private final Map<String, String> chineseRoomNames = new HashMap<>();
    private boolean chinese;

    public AgentVisualizerMod() {
        String configuredPath = System.getProperty("agent.overlay.state");
        if (configuredPath == null || configuredPath.trim().isEmpty()) {
            configuredPath = System.getenv("AGENT_OVERLAY_STATE");
        }
        snapshotPath = configuredPath == null || configuredPath.trim().isEmpty()
            ? null
            : Paths.get(configuredPath).toAbsolutePath();
        BaseMod.subscribe(this);
    }

    public static void initialize() {
        new AgentVisualizerMod();
    }

    @Override
    public void receivePostInitialize() {
        chinese = Settings.language == Settings.GameLanguage.ZHS
            || Settings.language == Settings.GameLanguage.ZHT;
        if (chinese) {
            loadChineseNames();
        }
        createRenderResources();
        if (snapshotPath != null) {
            Thread watcher = new Thread(
                this::watchSnapshot,
                "agent-overlay-watcher"
            );
            watcher.setDaemon(true);
            watcher.start();
        }
    }

    private void loadChineseNames() {
        try (InputStream stream = AgentVisualizerMod.class
            .getResourceAsStream(
                "/va/visualizer/translations_zh_CN.json"
            )) {
            if (stream == null) {
                throw new IllegalStateException(
                    "missing visualizer translation resource"
                );
            }
            JsonValue translations = new JsonReader().parse(stream);
            loadLocalizedNames(translations.get("rooms"), chineseRoomNames);
            loadLocalizedNames(translations.get("cards"), chineseCardNames);
            loadLocalizedNames(
                translations.get("monsters"),
                chineseMonsterNames
            );
        } catch (Exception ignored) {
            chineseCardNames.clear();
            chineseMonsterNames.clear();
            chineseRoomNames.clear();
        }
    }

    private static void loadLocalizedNames(
        JsonValue source,
        Map<String, String> destination
    ) {
        if (source == null || !source.isObject()) {
            throw new IllegalArgumentException("invalid translation section");
        }
        for (JsonValue item = source.child; item != null; item = item.next) {
            destination.put(
                item.name.toLowerCase(Locale.ROOT),
                item.asString()
            );
        }
    }

    private void createRenderResources() {
        float scale = Math.max(0.75f, Settings.scale);
        titleFontGenerator = new FreeTypeFontGenerator(
            Gdx.files.internal("font/zhs/NotoSansMonoCJKsc-Regular.otf")
        );
        bodyFontGenerator = new FreeTypeFontGenerator(
            Gdx.files.internal("font/zhs/NotoSansMonoCJKsc-Regular.otf")
        );
        titleFont = titleFontGenerator.generateFont(fontParameters(
            Math.max(15, Math.round(17.0f * scale)),
            Math.max(0.35f, scale * 0.42f),
            0,
            Math.max(0, Math.round(scale * 0.6f))
        ));
        bodyFont = bodyFontGenerator.generateFont(fontParameters(
            Math.max(12, Math.round(13.0f * scale)),
            Math.max(0.25f, scale * 0.30f),
            -1,
            0
        ));

        Pixmap source = new Pixmap(1, 1, Pixmap.Format.RGBA8888);
        source.setColor(Color.WHITE);
        source.fill();
        pixel = new Texture(source);
        source.dispose();
    }

    private static FreeTypeFontGenerator.FreeTypeFontParameter fontParameters(
        int size,
        float borderWidth,
        int spaceY,
        int spaceX
    ) {
        FreeTypeFontGenerator.FreeTypeFontParameter parameters =
            new FreeTypeFontGenerator.FreeTypeFontParameter();
        parameters.size = size;
        parameters.incremental = true;
        parameters.borderWidth = borderWidth;
        parameters.borderColor = new Color(0.01f, 0.025f, 0.04f, 0.82f);
        parameters.color = Color.WHITE;
        parameters.spaceY = spaceY;
        parameters.spaceX = spaceX;
        return parameters;
    }

    private void watchSnapshot() {
        FileTime lastModified = null;
        while (!Thread.currentThread().isInterrupted()) {
            try {
                if (Files.isRegularFile(snapshotPath)) {
                    FileTime modified = Files.getLastModifiedTime(snapshotPath);
                    if (!modified.equals(lastModified)) {
                        long size = Files.size(snapshotPath);
                        if (size > 0 && size <= MAX_SNAPSHOT_BYTES) {
                            OverlayState next = GSON.fromJson(
                                new String(
                                    Files.readAllBytes(snapshotPath),
                                    StandardCharsets.UTF_8
                                ),
                                OverlayState.class
                            );
                            if (next != null && next.schemaVersion == 4) {
                                next.normalize();
                                state = next;
                            }
                        }
                        lastModified = modified;
                    }
                }
                Thread.sleep(100L);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            } catch (Exception ignored) {
                // A display file must never affect the game process.
            }
        }
    }

    @Override
    public void receivePostRender(SpriteBatch spriteBatch) {
        OverlayState current = state;
        if (
            current == null
            || titleFont == null
            || bodyFont == null
            || pixel == null
        ) {
            return;
        }
        float scale = Math.max(0.75f, Settings.scale);
        float margin = 16.0f * scale;
        float gap = 8.0f * scale;
        float leftTop = Settings.HEIGHT - 154.0f * scale;
        float rightTop = leftTop;
        float leftBottom = margin;
        float rightBottom = Math.max(
            135.0f * scale,
            Settings.HEIGHT * 0.17f
        );
        float referenceHeight = rightTop - rightBottom;
        float leftAvailableHeight = leftTop - leftBottom;
        float leftWidth = Math.min(430.0f * scale, Settings.WIDTH * 0.34f);
        float rightWidth = Math.min(340.0f * scale, Settings.WIDTH * 0.25f);

        float mapHeight = mapContentHeight(current, leftWidth, scale);
        float buildHeight = buildContentHeight(current, leftWidth, scale);
        float strategyHeight = Math.min(
            leftAvailableHeight * 0.16f,
            strategyContentHeight(current, leftWidth, scale)
        );
        float mctsMaxHeight = Math.max(
            0.0f,
            leftAvailableHeight
                - mapHeight
                - buildHeight
                - strategyHeight
                - gap * 3
        );
        float mctsHeight = current.mctsPanel.actions.isEmpty()
            ? 0.0f
            : Math.min(
                mctsMaxHeight,
                mctsContentHeight(current, leftWidth, scale)
            );
        float actionHeight = Math.min(
            referenceHeight,
            actionContentHeight(current, rightWidth, scale)
        );

        float mapY = leftTop - mapHeight;
        float buildY = mapY - gap - buildHeight;
        float strategyY = buildY - gap - strategyHeight;
        float mctsY = strategyY - gap - mctsHeight;
        float rightX = Settings.WIDTH - rightWidth - margin;
        float actionY = rightTop - actionHeight;

        drawMapPanel(
            spriteBatch,
            current,
            margin,
            mapY,
            leftWidth,
            mapHeight,
            scale
        );
        drawBuildPanel(
            spriteBatch,
            current,
            margin,
            buildY,
            leftWidth,
            buildHeight,
            scale
        );
        drawStrategyPanel(
            spriteBatch,
            current,
            margin,
            strategyY,
            leftWidth,
            strategyHeight,
            scale
        );
        if (mctsHeight > 0.0f) {
            drawMctsPanel(
                spriteBatch,
                current,
                margin,
                mctsY,
                leftWidth,
                mctsHeight,
                scale
            );
        }
        drawActionPanel(
            spriteBatch,
            current,
            rightX,
            actionY,
            rightWidth,
            actionHeight,
            scale
        );
    }

    private float mapContentHeight(
        OverlayState state,
        float width,
        float scale
    ) {
        float textWidth = width - 26.0f * scale;
        return 24.0f * scale
            + renderedTextHeight(titleFont, text("Route", "路线"), textWidth)
            + 3.0f * scale
            + routeTextHeight(state, textWidth, scale);
    }

    private float strategyContentHeight(
        OverlayState state,
        float width,
        float scale
    ) {
        float textWidth = width - 26.0f * scale;
        return 24.0f * scale
            + renderedTextHeight(titleFont, text("Strategy", "策略"), textWidth)
            + 3.0f * scale
            + renderedTextHeight(
                titleFont,
                strategyName(state.strategyPanel),
                textWidth
            );
    }

    private float buildContentHeight(
        OverlayState state,
        float width,
        float scale
    ) {
        float textWidth = width - 26.0f * scale;
        float chipHeight = bodyFont.getCapHeight() + 8.0f * scale;
        float rowAdvance = chipHeight + 4.0f * scale;
        int cardCount = displayCards(state.buildPanel.cards).size();
        int rows = (cardCount + BUILD_COLUMNS - 1) / BUILD_COLUMNS;
        float chipsHeight = rows == 0
            ? 0.0f
            : chipHeight + (rows - 1) * rowAdvance;
        return 24.0f * scale
            + renderedTextHeight(
                titleFont,
                buildTitle(state.buildPanel),
                textWidth
            )
            + 5.0f * scale
            + chipsHeight;
    }

    private float mctsContentHeight(
        OverlayState state,
        float width,
        float scale
    ) {
        float textWidth = width - 26.0f * scale;
        int rows = 1 + state.mctsPanel.actions.size();
        float rowHeight = bodyFont.getLineHeight() + 4.0f * scale;
        return 24.0f * scale
            + renderedTextHeight(titleFont, text("Combat", "战斗"), textWidth)
            + 4.0f * scale
            + rows * rowHeight;
    }

    private float actionContentHeight(
        OverlayState state,
        float width,
        float scale
    ) {
        float textWidth = width - 26.0f * scale;
        ActionPanel action = state.actionPanel;
        String actionLabel = actionLabel(action);
        String history = safe(action.history);
        float height = 33.0f * scale
            + renderedTextHeight(titleFont, text("Action", "行动"), textWidth);
        if (actionLabel.isEmpty() && history.isEmpty()) {
            return height;
        }

        String meta = safe(action.context);
        if (!meta.isEmpty()) {
            height += renderedTextHeight(bodyFont, meta, textWidth)
                + 5.0f * scale;
        }
        if (!actionLabel.isEmpty()) {
            height += renderedTextHeight(
                bodyFont,
                text("Action", "行动") + "  " + actionLabel,
                textWidth
            ) + 4.0f * scale;
        }
        if (!history.isEmpty()) {
            height += renderedTextHeight(bodyFont, history, textWidth);
        }
        return height;
    }

    private float renderedTextHeight(
        BitmapFont font,
        String text,
        float width
    ) {
        return Math.max(textHeight(font, text, width), font.getLineHeight());
    }

    private List<RouteDisplayItem> routeItems(OverlayState state) {
        List<RouteDisplayItem> items = new ArrayList<>();
        for (RouteRoom room : state.mapPanel.rooms) {
            items.add(new RouteDisplayItem(
                "F" + number(room.floor) + " "
                    + localizedName(chineseRoomNames, safe(room.name)),
                room.current
            ));
        }
        if (!safe(state.mapPanel.boss).isEmpty()) {
            StringBuilder boss = new StringBuilder();
            if (state.mapPanel.bossFloor != null) {
                boss.append("F")
                    .append(number(state.mapPanel.bossFloor))
                    .append(" ");
            }
            boss.append(text("Boss", "首领"))
                .append(" ")
                .append(localizedMonster(state.mapPanel.boss));
            items.add(new RouteDisplayItem(
                boss.toString(),
                state.mapPanel.bossCurrent
                    || (
                        state.run.floor != null
                        && state.mapPanel.bossFloor != null
                        && state.run.floor.doubleValue()
                            == state.mapPanel.bossFloor.doubleValue()
                    )
            ));
        }
        return items;
    }

    private float routeTextHeight(
        OverlayState state,
        float width,
        float scale
    ) {
        List<RouteDisplayItem> items = routeItems(state);
        if (items.isEmpty()) {
            return bodyFont.getLineHeight();
        }
        float arrowWidth = singleLineWidth(bodyFont, "  →  ");
        float usedWidth = 0.0f;
        int rows = 1;
        for (RouteDisplayItem item : items) {
            float itemWidth = Math.min(
                width,
                singleLineWidth(bodyFont, item.label)
            );
            float required = itemWidth + (usedWidth > 0.0f ? arrowWidth : 0.0f);
            if (usedWidth > 0.0f && usedWidth + required > width) {
                rows += 1;
                usedWidth = itemWidth;
            } else {
                usedWidth += required;
            }
        }
        return rows * (bodyFont.getLineHeight() + 4.0f * scale)
            - 4.0f * scale;
    }

    private String strategyName(StrategyPanel strategy) {
        String name = safe(strategy.name);
        if (name.isEmpty() || "NONE".equalsIgnoreCase(name)) {
            return text("Not established", "尚未形成");
        }
        return name;
    }

    private String buildTitle(BuildPanel build) {
        return text("Deck", "卡组") + "  ·  " + build.cardCount;
    }

    private String localizedCardLabel(CardRow card) {
        String rawName = safe(card.name);
        StringBuilder label = new StringBuilder(
            localizedName(chineseCardNames, rawName)
        );
        if (card.upgrades == 1) {
            label.append("+");
        } else if (card.upgrades > 1) {
            label.append("+").append(card.upgrades);
        }
        if (card.count > 1) {
            label.append(" ×").append(card.count);
        }
        return label.toString();
    }

    private static List<CardRow> displayCards(List<CardRow> cards) {
        List<CardRow> rows = new ArrayList<>();
        for (CardRow card : cards) {
            int upgraded = Math.min(card.count, Math.max(0, card.upgrades));
            addCardRow(rows, card.name, card.count - upgraded, 0);
            addCardRow(rows, card.name, upgraded, 1);
        }
        return rows;
    }

    private static void addCardRow(
        List<CardRow> rows,
        String name,
        int count,
        int upgrades
    ) {
        if (count <= 0) {
            return;
        }
        CardRow row = new CardRow();
        row.name = name;
        row.count = count;
        row.upgrades = upgrades;
        rows.add(row);
    }

    private String localizedBattleCard(String value) {
        String raw = safe(value);
        int plus = raw.lastIndexOf('+');
        String base = plus > 0 ? raw.substring(0, plus).trim() : raw;
        String suffix = plus > 0 ? raw.substring(plus) : "";
        String translated = localizedName(chineseCardNames, base);
        return translated.equals(base) ? raw : translated + suffix;
    }

    private String localizedMonster(String value) {
        return localizedName(chineseMonsterNames, safe(value));
    }

    private String localizedName(
        Map<String, String> names,
        String value
    ) {
        if (!chinese) {
            return value;
        }
        String translated = names.get(value.toLowerCase(Locale.ROOT));
        return translated == null || translated.isEmpty() ? value : translated;
    }

    private String text(String english, String chineseText) {
        return chinese ? chineseText : english;
    }

    private String actionLabel(ActionPanel action) {
        String value = safe(action.action);
        if ("Thinking ...".equals(value)) {
            return text(value, "思考中 ...");
        }
        if ("LLM failed".equals(value)) {
            return text(value, "LLM 失败");
        }
        return value;
    }

    private static String mctsWinRate(RootAction action) {
        return action.winRate == null
            ? "—"
            : String.format(
                Locale.ROOT,
                "%.0f%%",
                action.winRate * 100.0
            );
    }

    private static String mctsHp(RootAction action) {
        return action.endHp == null
            ? "—"
            : Long.toString(Math.round(action.endHp));
    }

    private void drawMapPanel(
        SpriteBatch batch,
        OverlayState state,
        float x,
        float y,
        float width,
        float height,
        float scale
    ) {
        panel(batch, x, y, width, height);
        float padding = 13.0f * scale;
        float textWidth = width - padding * 2;
        float cursor = y + height - 14.0f * scale;
        cursor = drawText(
            batch,
            titleFont,
            text("Route", "路线"),
            x + padding,
            cursor,
            textWidth,
            TITLE,
            3.0f * scale
        );

        drawRoute(
            batch,
            state,
            x + padding,
            cursor,
            textWidth,
            scale
        );
    }

    private void drawRoute(
        SpriteBatch batch,
        OverlayState state,
        float x,
        float y,
        float width,
        float scale
    ) {
        List<RouteDisplayItem> items = routeItems(state);
        if (items.isEmpty()) {
            drawSingleLine(
                batch,
                bodyFont,
                "—",
                x,
                y,
                width,
                Align.left,
                MUTED
            );
            return;
        }
        String arrow = "  →  ";
        float arrowWidth = singleLineWidth(bodyFont, arrow);
        float lineAdvance = bodyFont.getLineHeight() + 4.0f * scale;
        float currentX = x;
        float baseline = y;
        float right = x + width;
        for (RouteDisplayItem item : items) {
            float itemWidth = Math.min(
                width,
                singleLineWidth(bodyFont, item.label)
            );
            float prefixWidth = currentX > x ? arrowWidth : 0.0f;
            if (
                currentX > x
                && currentX + prefixWidth + itemWidth > right
            ) {
                currentX = x;
                baseline -= lineAdvance;
                prefixWidth = 0.0f;
            }
            if (prefixWidth > 0.0f) {
                drawSingleLine(
                    batch,
                    bodyFont,
                    arrow,
                    currentX,
                    baseline,
                    arrowWidth,
                    Align.left,
                    MUTED
                );
                currentX += arrowWidth;
            }
            if (item.current) {
                fill(
                    batch,
                    ROUTE_CURRENT_BACKGROUND,
                    currentX - 3.0f * scale,
                    baseline - bodyFont.getCapHeight() - 4.0f * scale,
                    itemWidth + 6.0f * scale,
                    bodyFont.getLineHeight() + 4.0f * scale
                );
                fill(
                    batch,
                    ACCENT,
                    currentX - 3.0f * scale,
                    baseline - bodyFont.getCapHeight() - 4.0f * scale,
                    Math.max(1.0f, 1.0f * scale),
                    bodyFont.getLineHeight() + 4.0f * scale
                );
            }
            drawSingleLine(
                batch,
                bodyFont,
                item.label,
                currentX,
                baseline,
                itemWidth,
                Align.left,
                item.current ? GOOD : TEXT
            );
            currentX += itemWidth;
        }
    }

    private void drawStrategyPanel(
        SpriteBatch batch,
        OverlayState state,
        float x,
        float y,
        float width,
        float height,
        float scale
    ) {
        panel(batch, x, y, width, height);
        float padding = 13.0f * scale;
        float textWidth = width - padding * 2;
        float cursor = y + height - 14.0f * scale;
        StrategyPanel strategy = state.strategyPanel;
        Color lineColor = strategyStatusColor(strategy.status);
        cursor = drawText(
            batch,
            titleFont,
            text("Strategy", "策略"),
            x + padding,
            cursor,
            textWidth,
            TITLE,
            3.0f * scale
        );
        drawText(
            batch,
            titleFont,
            strategyName(strategy),
            x + padding,
            cursor,
            textWidth,
            lineColor,
            0.0f
        );
    }

    private void drawMctsPanel(
        SpriteBatch batch,
        OverlayState state,
        float x,
        float y,
        float width,
        float height,
        float scale
    ) {
        MctsPanel mcts = state.mctsPanel;
        if (mcts.actions.isEmpty()) {
            return;
        }
        panel(batch, x, y, width, height);
        float padding = 13.0f * scale;
        float textWidth = width - padding * 2;
        float cursor = y + height - 14.0f * scale;
        cursor = drawText(
            batch,
            titleFont,
            text("Combat", "战斗"),
            x + padding,
            cursor,
            textWidth,
            TITLE,
            4.0f * scale
        );
        cursor = drawMctsRow(
            batch,
            text("Card", "卡牌"),
            text("Target", "对象"),
            text("Win Rate", "胜率"),
            "HP",
            x + padding,
            cursor,
            textWidth,
            MUTED,
            scale
        );
        for (RootAction action : mcts.actions) {
            if (cursor <= y + 13.0f * scale) {
                break;
            }
            cursor = drawMctsRow(
                batch,
                localizedBattleCard(action.label),
                localizedMonster(action.target),
                mctsWinRate(action),
                mctsHp(action),
                x + padding,
                cursor,
                textWidth,
                action.selected ? GOOD : TEXT,
                scale
            );
        }
    }

    private void drawBuildPanel(
        SpriteBatch batch,
        OverlayState state,
        float x,
        float y,
        float width,
        float height,
        float scale
    ) {
        panel(batch, x, y, width, height);
        float padding = 13.0f * scale;
        float textWidth = width - padding * 2;
        float cursor = y + height - 14.0f * scale;
        BuildPanel build = state.buildPanel;
        List<CardRow> cards = displayCards(build.cards);
        cursor = drawText(
            batch,
            titleFont,
            buildTitle(build),
            x + padding,
            cursor,
            textWidth,
            TITLE,
            4.0f * scale
        );

        float chipPadding = 6.0f * scale;
        float chipGap = 5.0f * scale;
        float chipHeight = bodyFont.getCapHeight() + 8.0f * scale;
        float rowAdvance = chipHeight + 4.0f * scale;
        float chipWidth = (
            textWidth - chipGap * (BUILD_COLUMNS - 1)
        ) / BUILD_COLUMNS;
        float firstBaseline = cursor - 1.0f * scale;
        for (int index = 0; index < cards.size(); index++) {
            CardRow card = cards.get(index);
            String label = localizedCardLabel(card);
            int column = index % BUILD_COLUMNS;
            int row = index / BUILD_COLUMNS;
            float currentX = x + padding + column * (chipWidth + chipGap);
            float baseline = firstBaseline - row * rowAdvance;
            if (baseline - chipHeight < y + 7.0f * scale) {
                break;
            }
            fill(
                batch,
                CHIP_BACKGROUND,
                currentX,
                baseline - bodyFont.getCapHeight() - 4.0f * scale,
                chipWidth,
                chipHeight
            );
            outline(
                batch,
                CHIP_BORDER,
                currentX,
                baseline - bodyFont.getCapHeight() - 4.0f * scale,
                chipWidth,
                chipHeight,
                Math.max(1.0f, 0.65f * scale)
            );
            drawSingleLine(
                batch,
                bodyFont,
                label,
                currentX + chipPadding,
                baseline,
                chipWidth - chipPadding * 2,
                Align.left,
                card.upgrades > 0 ? GOOD : TEXT
            );
        }
    }

    private void drawActionPanel(
        SpriteBatch batch,
        OverlayState state,
        float x,
        float y,
        float width,
        float height,
        float scale
    ) {
        panel(batch, x, y, width, height);
        float padding = 13.0f * scale;
        float textWidth = width - padding * 2;
        float cursor = y + height - 14.0f * scale;
        ActionPanel action = state.actionPanel;
        cursor = drawText(
            batch,
            titleFont,
            text("Action", "行动"),
            x + padding,
            cursor,
            textWidth,
            TITLE,
            6.0f * scale
        );
        String actionLabel = actionLabel(action);
        String history = safe(action.history);
        if (actionLabel.isEmpty() && history.isEmpty()) {
            return;
        }
        String meta = safe(action.context);
        if (!meta.isEmpty()) {
            cursor = drawText(
                batch,
                bodyFont,
                meta,
                x + padding,
                cursor,
                textWidth,
                ACCENT,
                5.0f * scale
            );
        }
        if (!actionLabel.isEmpty()) {
            cursor = drawText(
                batch,
                bodyFont,
                text("Action", "行动") + "  " + actionLabel,
                x + padding,
                cursor,
                textWidth,
                GOOD,
                4.0f * scale
            );
        }
        if (history.isEmpty()) {
            return;
        }
        float available = Math.max(20.0f * scale, cursor - y - padding);
        String visible = tailFittingHeight(
            bodyFont,
            history,
            textWidth,
            available
        );
        drawText(
            batch,
            bodyFont,
            visible,
            x + padding,
            cursor,
            textWidth,
            TEXT,
            0.0f
        );
    }

    private void panel(
        SpriteBatch batch,
        float x,
        float y,
        float width,
        float height
    ) {
        float scale = Math.max(0.75f, Settings.scale);
        float hairline = Math.max(1.0f, 0.65f * scale);
        float accentWidth = Math.min(width * 0.22f, 58.0f * scale);
        float secondaryWidth = Math.min(width * 0.05f, 12.0f * scale);
        float headerHeight = Math.min(height, 28.0f * scale);

        fill(
            batch,
            PANEL_SHADOW,
            x + 2.0f * scale,
            y - 2.0f * scale,
            width,
            height
        );
        fill(batch, PANEL_BACKGROUND, x, y, width, height);
        fill(
            batch,
            PANEL_HEADER_WASH,
            x,
            y + height - headerHeight,
            width,
            headerHeight
        );
        outline(
            batch,
            PANEL_BORDER_SOFT,
            x,
            y,
            width,
            height,
            hairline
        );

        fill(
            batch,
            ACCENT_GLOW,
            x,
            y + height - hairline * 3.0f,
            accentWidth + 8.0f * scale,
            hairline * 3.0f
        );
        fill(
            batch,
            ACCENT,
            x,
            y + height - hairline,
            accentWidth,
            hairline
        );
        fill(
            batch,
            ACCENT,
            x,
            y + height - 10.0f * scale,
            hairline,
            10.0f * scale
        );
        fill(
            batch,
            ACCENT_SECONDARY,
            x + width - secondaryWidth,
            y + height - hairline,
            secondaryWidth,
            hairline
        );
        fill(
            batch,
            PANEL_BORDER,
            x + width - 18.0f * scale,
            y,
            18.0f * scale,
            hairline
        );
        fill(
            batch,
            PANEL_BORDER,
            x + width - hairline,
            y,
            hairline,
            7.0f * scale
        );
    }

    private void outline(
        SpriteBatch batch,
        Color color,
        float x,
        float y,
        float width,
        float height,
        float thickness
    ) {
        fill(batch, color, x, y, width, thickness);
        fill(batch, color, x, y + height - thickness, width, thickness);
        fill(batch, color, x, y, thickness, height);
        fill(batch, color, x + width - thickness, y, thickness, height);
    }

    private void fill(
        SpriteBatch batch,
        Color color,
        float x,
        float y,
        float width,
        float height
    ) {
        Color original = batch.getColor();
        float originalR = original.r;
        float originalG = original.g;
        float originalB = original.b;
        float originalA = original.a;
        batch.setColor(color);
        batch.draw(pixel, x, y, width, height);
        batch.setColor(originalR, originalG, originalB, originalA);
    }

    private float drawText(
        SpriteBatch batch,
        BitmapFont font,
        String text,
        float x,
        float y,
        float width,
        Color color,
        float gap
    ) {
        glyphLayout.setText(
            font,
            safe(text),
            color,
            width,
            Align.left,
            true
        );
        font.draw(batch, glyphLayout, x, y);
        float occupiedHeight = Math.max(
            glyphLayout.height,
            font.getLineHeight()
        );
        return y - occupiedHeight - gap;
    }

    private float drawMctsRow(
        SpriteBatch batch,
        String card,
        String target,
        String winRate,
        String hp,
        float x,
        float y,
        float width,
        Color color,
        float scale
    ) {
        float columnWidth = width / 4.0f;
        drawSingleLine(
            batch,
            bodyFont,
            card,
            x,
            y,
            columnWidth,
            Align.left,
            color
        );
        drawSingleLine(
            batch,
            bodyFont,
            target,
            x + columnWidth,
            y,
            columnWidth,
            Align.center,
            color
        );
        drawSingleLine(
            batch,
            bodyFont,
            winRate,
            x + columnWidth * 2.0f,
            y,
            columnWidth,
            Align.center,
            color
        );
        drawSingleLine(
            batch,
            bodyFont,
            hp,
            x + columnWidth * 3.0f,
            y,
            columnWidth,
            Align.center,
            color
        );
        return y - bodyFont.getLineHeight() - 4.0f * scale;
    }

    private void drawSingleLine(
        SpriteBatch batch,
        BitmapFont font,
        String text,
        float x,
        float y,
        float width,
        int align,
        Color color
    ) {
        String value = safe(text);
        glyphLayout.setText(
            font,
            value,
            0,
            value.length(),
            color,
            width,
            align,
            false,
            "…"
        );
        font.draw(batch, glyphLayout, x, y);
    }

    private static Color strategyStatusColor(String status) {
        String normalized = safe(status).toUpperCase(Locale.ROOT);
        if ("ONLINE".equals(normalized)) {
            return GOOD;
        }
        if ("COMMITTED".equals(normalized)) {
            return ACCENT;
        }
        return WARN;
    }

    private static String safe(String value) {
        return value == null ? "" : value;
    }

    private static String number(Double value) {
        if (value == null) {
            return "?";
        }
        if (Math.rint(value) == value) {
            return Long.toString(Math.round(value));
        }
        return value.toString();
    }

    private String tailFittingHeight(
        BitmapFont font,
        String text,
        float width,
        float maxHeight
    ) {
        String normalized = safe(text);
        if (textHeight(font, normalized, width) <= maxHeight) {
            return normalized;
        }
        int codePoints = normalized.codePointCount(0, normalized.length());
        int low = 0;
        int high = codePoints;
        while (low < high) {
            int candidateLength = (low + high + 1) / 2;
            int start = normalized.offsetByCodePoints(
                0,
                codePoints - candidateLength
            );
            String candidate = "…" + normalized.substring(start);
            if (textHeight(font, candidate, width) <= maxHeight) {
                low = candidateLength;
            } else {
                high = candidateLength - 1;
            }
        }
        int start = normalized.offsetByCodePoints(0, codePoints - low);
        return "…" + normalized.substring(start);
    }

    private float textHeight(
        BitmapFont font,
        String text,
        float width
    ) {
        glyphLayout.setText(
            font,
            safe(text),
            Color.WHITE,
            width,
            Align.left,
            true
        );
        return glyphLayout.height;
    }

    private float singleLineWidth(BitmapFont font, String text) {
        glyphLayout.setText(font, safe(text));
        return glyphLayout.width;
    }

    private static final class OverlayState {
        int schemaVersion;
        long sequence;
        RunState run;
        MapPanel mapPanel;
        StrategyPanel strategyPanel;
        MctsPanel mctsPanel;
        BuildPanel buildPanel;
        ActionPanel actionPanel;

        void normalize() {
            if (run == null) {
                run = new RunState();
            }
            if (mapPanel == null) {
                mapPanel = new MapPanel();
            }
            if (strategyPanel == null) {
                strategyPanel = new StrategyPanel();
            }
            if (mctsPanel == null) {
                mctsPanel = new MctsPanel();
            }
            if (buildPanel == null) {
                buildPanel = new BuildPanel();
            }
            if (actionPanel == null) {
                actionPanel = new ActionPanel();
            }
            mapPanel.normalize();
            mctsPanel.normalize();
            buildPanel.normalize();
        }
    }

    private static final class RunState {
        Double floor;
    }

    private static final class MapPanel {
        List<RouteRoom> rooms;
        String boss;
        Double bossFloor;
        boolean bossCurrent;

        void normalize() {
            if (rooms == null) {
                rooms = Collections.emptyList();
            }
        }
    }

    private static final class RouteRoom {
        Double floor;
        String name;
        boolean current;
    }

    private static final class RouteDisplayItem {
        final String label;
        final boolean current;

        RouteDisplayItem(String label, boolean current) {
            this.label = label;
            this.current = current;
        }
    }

    private static final class StrategyPanel {
        String status;
        String name;
    }

    private static final class MctsPanel {
        List<RootAction> actions;

        void normalize() {
            if (actions == null) {
                actions = Collections.emptyList();
            }
        }
    }

    private static final class BuildPanel {
        int cardCount;
        List<CardRow> cards;

        void normalize() {
            if (cards == null) {
                cards = Collections.emptyList();
            }
        }
    }

    private static final class CardRow {
        String name;
        int upgrades;
        int count;
    }

    private static final class ActionPanel {
        String context;
        String action;
        String history;
    }

    private static final class RootAction {
        String label;
        String target;
        boolean selected;
        Double winRate;
        Double endHp;
    }
}
