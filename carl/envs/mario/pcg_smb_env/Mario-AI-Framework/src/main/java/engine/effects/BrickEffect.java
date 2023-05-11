package engine.effects;

import java.awt.Graphics;

import engine.core.MarioEffect;

public class BrickEffect extends MarioEffect {

    public BrickEffect(float x, float y, float xv, float yv) {
        super(x, y, xv, yv, 0, 3, 16, 10);
    }

    @Override
    public void render(Graphics og, float cameraX, float cameraY) {
        this.graphics.index = this.startingIndex + this.life % 4;
        this.ya *= 0.95f;
        super.render(og, cameraX, cameraY);
    }

}