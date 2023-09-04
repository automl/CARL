package agents.andySloane;

public class EnemyState extends SpriteState {
    // we need to simulate enemy death, too, cuz it doesn't tell us whether
    // we're looking at dead enemies
    public boolean flyDeath = false;

    public static final float width = 4;

    @Override
    public final float height() {
        return type >= 4 && type <= 7 ? 24 : 12;
    }

    @Override
    public SpriteState clone() {
        EnemyState e = new EnemyState(x, y, type);
        e.xa = xa;
        e.ya = ya;
        e.facing = facing;
        e.deadTime = deadTime;
        e.flyDeath = flyDeath;
        e.onGround = onGround;
        return e;
    }

    public final boolean avoidCliffs() {
        return type == KIND_RED_KOOPA;
    }

    public final boolean winged() {
        switch (type) {
            case KIND_GOOMBA_WINGED:
            case KIND_RED_KOOPA_WINGED:
            case KIND_GREEN_KOOPA_WINGED:
            case KIND_SPIKY_WINGED:
                return true;
        }
        return false;
    }

    public final boolean spiky() {
        switch (type) {
            case KIND_FLOWER_ENEMY:
            case KIND_SPIKY:
            case KIND_SPIKY_WINGED:
                return true;
        }
        return false;
    }

    public final boolean noFireballDeath() {
        switch (type) {
            case KIND_SPIKY:
            case KIND_SPIKY_WINGED:
                return true;
        }
        return false;
    }

    @Override
    public final boolean dead() {
        return deadTime != 0;
    }

    EnemyState(float _x, float _y, int _type) {
        x = _x;
        y = _y;
        type = _type;
        // most likely they've been falling for a step before we ever see them
        if (winged())
            ya = 0.6f;
        else
            ya = 2;
        facing = -1;
    }

    // returns false iff we should remove the enemy from the list
    public boolean move(WorldState ws) {
        if (deadTime > 0) {
            deadTime--;

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

        float sideWaysSpeed = 1.75f;

        if (xa > 2)
            facing = 1;
        else if (xa < -2)
            facing = -1;

        xa = facing * sideWaysSpeed;

        if (!move(xa, 0, ws))
            facing = -facing;
        onGround = false;
        move(0, ya, ws);

        ya *= winged() ? 0.95f : 0.85f;
        xa *= DAMPING_X; // useless

        if (!onGround) {
            if (winged())
                ya += 0.6f;
            else
                ya += 2;
        } else if (winged())
            ya = -10;

        return true;
    }

    @Override
    public void resync(float x, float y, float prev_x, float prev_y) {
        this.x = x;
        this.y = y;
        this.xa = x - prev_x;
        if (this.xa == 0) {
            // the only way we could be not moving horizontally is if we're dead.
            // since we mispredicted this death, assume we've only been dead
            // one frame.
            deadTime = 9;
            return;
        }
        facing = (this.xa < 0) ? -1 : 1;
        this.ya = (y - prev_y) * (winged() ? 0.95f : 0.85f);
        if (xa != 0 && ya == 0) {
            // if we're moving along the ground, then we aren't dead from a
            // shell or something
            // deadTime = 0;
            // hm. we can't necessarily resurrect enemies like this.
        }
        if (!onGround) {
            if (winged())
                ya += 0.6f;
            else
                ya += 2;
        }
        // this is unlikely to be accurate on winged dudes but whatever
        else if (winged())
            ya = -10;

    }

    private boolean move(float xa, float ya, WorldState ws) {
        float height = this.height();

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

            if (avoidCliffs() && onGround && !ws.isBlocking((int) ((x + xa + width) / 16), (int) ((y) / 16 + 1), xa, 1))
                collide = true;
        }
        if (xa < 0) {
            if (isBlocking(x + xa - width, y + ya - height, xa, ya, ws))
                collide = true;
            if (isBlocking(x + xa - width, y + ya - height / 2, xa, ya, ws))
                collide = true;
            if (isBlocking(x + xa - width, y + ya, xa, ya, ws))
                collide = true;

            if (avoidCliffs() && onGround && !ws.isBlocking((int) ((x + xa - width) / 16), (int) ((y) / 16 + 1), xa, 1))
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
        EnemyState e = (EnemyState) clone();
        if (e.winged()) {
            e.type--;
            e.ya = 0;
        } else {
            e.deadTime = 10;

            if (type == KIND_RED_KOOPA || type == KIND_GREEN_KOOPA) {
                ws.addShell(x, y);
            }
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
                if ((!spiky()) && ms.ya > 0 && yMarioD <= 0 && (!ms.onGround || !ms.wasOnGround)) {
                    ws = ws.stomp(this, ms);
                } else {
                    ms.getHurt();
                }
            }
        }
        return ws;
    }

    /*
     * public SpriteState shellCollideCheck(ShellState shell) { if (deadTime != 0)
     * return this;
     *
     * float xD = shell.x - x; float yD = shell.y - y;
     *
     * if (xD > -16 && xD < 16) { if (yD > -height() && yD < shell.height()) { xa =
     * shell.facing * 2; ya = -5; flyDeath = true; if (spriteTemplate != null)
     * spriteTemplate.isDead = true; deadTime = 100; //winged = false; type --; hPic
     * = -hPic; yPicO = -yPicO + 16; return true; } } return false; }
     *
     * public boolean fireballCollideCheck(SpriteState fireball) { if (deadTime !=
     * 0) return false;
     *
     * float xD = fireball.x - x; float yD = fireball.y - y;
     *
     * if (xD > -16 && xD < 16) { if (yD > -height && yD < 8) { if (noFireballDeath)
     * return true;
     *
     * xa = fireball.facing * 2; ya = -5; flyDeath = true; if (spriteTemplate !=
     * null) spriteTemplate.isDead = true; deadTime = 100; //winged = false; type
     * --; hPic = -hPic; yPicO = -yPicO + 16; return true; } } return false; }
     *
     * public void bumpCheck(int xTile, int yTile) { if (deadTime != 0) return;
     *
     * if (x + width > xTile * 16 && x - width < xTile * 16 + 16 && yTile == (int)
     * ((y - 1) / 16)) { xa = -world.mario.facing * 2; ya = -5; flyDeath = true; if
     * (spriteTemplate != null) spriteTemplate.isDead = true; deadTime = 100;
     * //winged = false; type --; hPic = -hPic; yPicO = -yPicO + 16; } }
     */
}
