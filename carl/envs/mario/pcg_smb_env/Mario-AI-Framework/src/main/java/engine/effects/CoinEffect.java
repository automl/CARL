package engine.effects;

import java.awt.Graphics;

import engine.core.MarioEffect;

public class CoinEffect extends MarioEffect {
    public CoinEffect(float x, float y) {
        super(x, y, 0, -8f, 0, 1, 0, 16);
    }

    @Override
    public void render(Graphics og, float cameraX, float cameraY) {
        this.graphics.index = this.startingIndex + this.life & 3;
        super.render(og, cameraX, cameraY);
    }
}