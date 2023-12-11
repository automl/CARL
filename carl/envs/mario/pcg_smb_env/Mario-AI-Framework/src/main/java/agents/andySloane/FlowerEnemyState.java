package agents.andySloane;

public final class FlowerEnemyState extends EnemyState {
    // we need to simulate enemy death, too, cuz it doesn't tell us whether
    // we're looking at dead enemies
    public int jumpTime = 0;
    public int yStart = 200;

    public static final float width = 2;

    @Override
    public SpriteState clone() {
        FlowerEnemyState e = new FlowerEnemyState(x, y);
        e.xa = xa;
        e.ya = ya;
        e.deadTime = deadTime;
        return e;
    }

    FlowerEnemyState(float _x, float _y) {
        super(_x, _y, KIND_FLOWER_ENEMY);
        // compensate for initial movement before we see them
        // this is kind of approximate but close enough for the time being
        yStart = (int) (_y + 28);
        ya = -4.9049f;
        jumpTime = 3;
        facing = 0;
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

            // death animation
            x += xa;
            y += ya;
            ya *= 0.95;
            ya += 1;
            return true;
        }

        if (y >= yStart) {
            y = yStart;

            int xd = 25; // (int)(Math.abs(ms.x-x));
            jumpTime++;
            if (jumpTime > 40 && xd > 24) {
                ya = -8;
            } else {
                ya = 0;
            }
        } else {
            jumpTime = 0;
        }

        y += ya;
        ya *= 0.9;
        ya += 0.1f;
        return true;
    }

    public void resync(float x, float y, float prev_x, float prev_y) {
        this.x = x;
        this.y = y;
        this.xa = 0;
        this.ya = (y - prev_y) * 0.9f + 0.1f;
    }

}
