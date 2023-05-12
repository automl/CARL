package agents.spencerSchumann;

import engine.core.MarioForwardModel;

/**
 * @author Spencer Schumann
 */
public class MarioState {

    public float x;
    public float y;
    public float vx;
    public float vy;
    public int mode;
    public float height;
    public boolean onGround;
    public boolean mayJump;
    public int jumpTime;

    private boolean first = true;

    public void update(MarioForwardModel model) {
        mode = model.getMarioMode();
        if (mode > 0) {
            height = 24.0f;
        } else {
            height = 12.0f;
        }
        onGround = model.isMarioOnGround();
        mayJump = model.mayMarioJump();

        float[] pos = model.getMarioFloatPos();
        if (first) {
            first = false;
            x = pos[0];
            y = pos[1];
        }
        float newVx = pos[0] - x;
        float newVy = pos[1] - y;
        vx = newVx;
        vy = newVy;
        x = pos[0];
        y = pos[1];
    }

    @Override
    public MarioState clone() {
        MarioState m = new MarioState();
        m.x = x;
        m.y = y;
        m.vx = vx;
        m.vy = vy;
        m.first = first;
        m.mode = mode;
        m.height = height;
        m.onGround = onGround;
        m.mayJump = mayJump;
        m.jumpTime = jumpTime;
        return m;
    }

    public boolean equals(MarioState other) {
        return x == other.x &&
                y == other.y &&
                first == other.first &&
                mode == other.mode &&
                height == other.height &&
                jumpTime == other.jumpTime &&
                mayJump == other.mayJump &&
                onGround == other.onGround;
    }
}
