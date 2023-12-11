package agents.andySloane;

public class ShellState extends SpriteState {
    public boolean carried = false;
    public boolean onGround = false;
    public boolean flyDeath = false;

    public static final float width = 4;
    public static final float height = 12;

    @Override
    public final float height() {
        return 12;
    }

    @Override
    public SpriteState clone() {
        ShellState e = new ShellState(x, y, false);
        e.xa = xa;
        e.ya = ya;
        e.facing = facing;
        e.deadTime = deadTime;
        e.carried = carried;
        e.onGround = onGround;
        e.flyDeath = flyDeath;
        return e;
    }

    @Override
    public final boolean dead() {
        return deadTime != 0;
    }

    ShellState(float _x, float _y, boolean predicted) {
        x = _x;
        y = _y;
        type = KIND_SHELL;
        xa = 0;
        ya = predicted ? -5 : -2.25f;
        facing = 0;
    }

    // returns false iff we should remove the enemy from the list
    public boolean move(WorldState ws) {
        if (carried) {
            ws.checkShellCollide(this);
            return false;
        }

        if (deadTime > 0) {
            deadTime--;

            // wait this is stupid. this function NEVER RETURNS FALSE
            if (deadTime == 0) {
                deadTime = 1; // keep us marked dead even when the timer goes away
                return false;
            }

            // we have to keep track of dead enemies so we can keep sync with their position
            if (flyDeath) {
                x += xa;
                y += ya;
                ya *= 0.95;
                ya += 1;
            }
            return true;
        }

        if (xa > 2)
            facing = 1;
        else if (xa < -2)
            facing = -1;

        if (facing != 0)
            ws.checkShellCollide(this);

        if (!move(xa, 0, ws))
            facing = -facing;

        onGround = false;
        move(0, ya, ws);
        ya *= 0.85f;
        xa *= DAMPING_X;

        if (!onGround)
            ya += 2;

        return true;
    }

    @Override
    public void resync(float x, float y, float prev_x, float prev_y) {
        this.x = x;
        this.y = y;
        this.xa = x - prev_x;
        facing = this.xa == 0 ? 0 : (this.xa < 0) ? -1 : 1;
        this.ya = (y - prev_y) * 0.85f;
        if (!onGround)
            ya += 2;
    }

    // WOO LET'S COPY AND PASTE THIS SOME MORE!
    private boolean move(float xa, float ya, WorldState ws) {
        while (xa > 8) {
            if (!move(8, 0, ws))
                return false;
            xa -= 8;
        }
        while (xa < -8) {
            if (!move(-8, 0, ws))
                return false;
            xa += 8;
        }
        while (ya > 8) {
            if (!move(0, 8, ws))
                return false;
            ya -= 8;
        }
        while (ya < -8) {
            if (!move(0, -8, ws))
                return false;
            ya += 8;
        }

        boolean collide = false;
        if (ya > 0) {
            if (isBlocking(x + xa - width, y + ya, xa, 0, ws))
                collide = true;
            else if (isBlocking(x + xa + width, y + ya, xa, 0, ws))
                collide = true;
            else if (isBlocking(x + xa - width, y + ya + 1, xa, ya, ws))
                collide = true;
            else if (isBlocking(x + xa + width, y + ya + 1, xa, ya, ws))
                collide = true;
        }
        if (ya < 0) {
            if (isBlocking(x + xa, y + ya - height, xa, ya, ws))
                collide = true;
            else if (collide || isBlocking(x + xa - width, y + ya - height, xa, ya, ws))
                collide = true;
            else if (collide || isBlocking(x + xa + width, y + ya - height, xa, ya, ws))
                collide = true;
        }
        if (xa > 0) {
            if (isBlocking(x + xa + width, y + ya - height, xa, ya, ws))
                collide = true;
            if (isBlocking(x + xa + width, y + ya - height / 2, xa, ya, ws))
                collide = true;
            if (isBlocking(x + xa + width, y + ya, xa, ya, ws))
                collide = true;
        }
        if (xa < 0) {
            if (isBlocking(x + xa - width, y + ya - height, xa, ya, ws))
                collide = true;
            if (isBlocking(x + xa - width, y + ya - height / 2, xa, ya, ws))
                collide = true;
            if (isBlocking(x + xa - width, y + ya, xa, ya, ws))
                collide = true;
        }

        if (collide) {
            if (xa < 0) {
                x = (int) ((x - width) / 16) * 16 + width;
                this.xa = 0;
            }
            if (xa > 0) {
                x = (int) ((x + width) / 16 + 1) * 16 - width - 1;
                this.xa = 0;
            }
            if (ya < 0) {
                y = (int) ((y - height) / 16) * 16 + height;
                this.ya = 0;
            }
            if (ya > 0) {
                y = (int) (y / 16 + 1) * 16 - 1;
                onGround = true;
            }
            return false;
        } else {
            x += xa;
            y += ya;
            return true;
        }
    }

    private boolean isBlocking(float _x, float _y, float xa, float ya, WorldState ws) {
        int x = (int) (_x / 16);
        int y = (int) (_y / 16);
        if (x == (int) (this.x / 16) && y == (int) (this.y / 16))
            return false;

        return ws.isBlocking(x, y, xa, ya);
    }

    @Override
    public SpriteState stomp(WorldState ws, MarioState ms) {
        ShellState e = (ShellState) clone();
        if (facing != 0) {
            e.facing = 0;
            e.xa = 0;
        } else {
            e.facing = ms.facing;
        }
        return e;
    }

    @Override
    public WorldState collideCheck(WorldState ws, MarioState ms) {
        if (deadTime != 0)
            return ws;

        float xMarioD = ms.x - x;
        float yMarioD = ms.y - y;
        float height = this.height();
        if (xMarioD > -width * 2 - 4 && xMarioD < width * 2 + 4) {
            if (yMarioD > -height && yMarioD < ms.height()) {
                if (!spiky() && ms.ya > 0 && yMarioD <= 0 && (!ms.onGround || !ms.wasOnGround)) {
                    ws = ws.stomp(this, ms);
                } else {
                    if (facing != 0)
                        ms.getHurt();
                    else {
                        ws.kick(this);
                        facing = ms.facing;
                    }
                }
            }
        }
        return ws;
    }

    /*
     *
     * // if shells hit one another, they both go poof public boolean
     * shellCollideCheck(ShellState shell) { if (deadTime != 0) return false;
     *
     * float xD = shell.x - x; float yD = shell.y - y;
     *
     * if (xD > -16 && xD < 16) { if (yD > -height() && yD < shell.height) { xa =
     * shell.facing * 2; ya = -5; flyDeath = true; deadTime = 100; return true; } }
     * return false; }
     *
     * public SpriteState fireballCollideCheck(SpriteState fireball) { if (deadTime
     * != 0) return false;
     *
     * float xD = fireball.x - x; float yD = fireball.y - y;
     *
     * if (xD > -16 && xD < 16) { if (yD > -height && yD < 8) { xa = fireball.facing
     * * 2; ya = -5; flyDeath = true; deadTime = 100; return true; } } return false;
     * }
     *
     * public SpriteState bumpCheck(int xTile, int yTile, MarioState ms) { if
     * (deadTime != 0) return;
     *
     * if (x + width > xTile * 16 && x - width < xTile * 16 + 16 && yTile == (int)
     * ((y - 1) / 16)) { xa = -ms.facing * 2; ya = -5; flyDeath = true; deadTime =
     * 100; } }
     */

}
