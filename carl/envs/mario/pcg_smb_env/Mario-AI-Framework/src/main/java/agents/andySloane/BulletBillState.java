package agents.andySloane;

public final class BulletBillState extends SpriteState {
    public static final float width = 4;

    @Override
    public final float height() {
        return 12;
    }

    @Override
    public final boolean dead() {
        return deadTime != 0;
    }

    @Override
    public SpriteState clone() {
        BulletBillState e = new BulletBillState(x, y, facing);
        e.xa = xa;
        e.ya = ya;
        e.facing = facing;
        e.deadTime = deadTime;
        return e;
    }

    BulletBillState(float _x, float _y, int dir) {
        x = _x;
        y = _y;
        type = KIND_BULLET_BILL;
        facing = dir;
        xa = 4 * facing;
    }

    BulletBillState(float _x, float _y, MarioState ms) {
        x = _x;
        y = _y;
        type = KIND_BULLET_BILL;
        facing = _x < ms.x ? 1 : -1;
        xa = 4 * facing;
    }

    // returns false iff we should remove the enemy from the list
    @Override
    public boolean move(WorldState ws) {
        if (deadTime > 0) {
            deadTime--;

            if (deadTime == 0) {
                deadTime = 1;
                return false;
            }

            // track our path as we fall, dead
            x += xa;
            y += ya;
            ya *= 0.95;
            ya += 1;
            return true;
        }

        float sideWaysSpeed = 4f;

        xa = facing * sideWaysSpeed;
        x += xa;

        return true;
    }

    @Override
    public SpriteState stomp(WorldState ws, MarioState ms) {
        BulletBillState e = (BulletBillState) clone();
        e.xa = 0;
        e.ya = 1;
        e.deadTime = 100;
        return e;
    }

    @Override
    public WorldState collideCheck(WorldState ws, MarioState ms) {
        if (deadTime != 0)
            return ws;

        float xMarioD = ms.x - x;
        float yMarioD = ms.y - y;
        // float w = 16;
        if (xMarioD > -16 && xMarioD < 16) {
            if (yMarioD > -height() && yMarioD < ms.height()) {
                if (ms.ya > 0 && yMarioD <= 0 && (!ms.onGround || !ms.wasOnGround)) {
                    ws = ws.stomp(this, ms);
                } else {
                    ms.getHurt();
                }
            }
        }
        return ws;
    }

    /*
     *
     * // TODO public boolean shellCollideCheck(Shell shell) { }
     *
     * public boolean fireballCollideCheck(SpriteState fireball) { }
     *
     */

}
