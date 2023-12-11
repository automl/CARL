package engine.effects;

import engine.core.MarioEffect;

public class DeathEffect extends MarioEffect {
    public DeathEffect(float x, float y, boolean flipX, int startIndex, float yv) {
        super(x, y, 0, yv, 0, 1f, startIndex, 30);
        this.graphics.flipX = flipX;
    }
}
