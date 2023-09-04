package engine.sprites;

import java.awt.Graphics;

import engine.core.MarioSprite;
import engine.graphics.MarioImage;
import engine.helper.Assets;
import engine.helper.EventType;
import engine.helper.SpriteType;

public class LifeMushroom extends MarioSprite {
    private boolean onGround = false;
    private int life;
    private MarioImage graphics;

    private static final float GROUND_INERTIA = 0.89f;
    private static final float AIR_INERTIA = 0.89f;

    public LifeMushroom(boolean visuals, float x, float y) {
        super(x, y, SpriteType.LIFE_MUSHROOM);
        this.width = 4;
        this.height = 12;
        this.facing = 1;
        this.life = 0;

        if (visuals) {
            this.graphics = new MarioImage(Assets.items, 3);
            this.graphics.width = 16;
            this.graphics.height = 16;
            this.graphics.originX = 8;
            this.graphics.originY = 15;
        }
    }

    @Override
    public MarioSprite clone() {
        LifeMushroom m = new LifeMushroom(false, this.x, this.y);
        m.xa = this.xa;
        m.ya = this.ya;
        m.initialCode = this.initialCode;
        m.width = this.width;
        m.height = this.height;
        m.facing = this.facing;
        m.life = this.life;
        m.onGround = this.onGround;
        return m;
    }

    public void collideCheck() {
        if (!this.alive) {
            return;
        }

        float xMarioD = world.mario.x - x;
        float yMarioD = world.mario.y - y;
        if (xMarioD > -16 && xMarioD < 16) {
            if (yMarioD > -height && yMarioD < world.mario.height) {
                world.addEvent(EventType.COLLECT, this.type.getValue());
                world.mario.collect1Up();
                world.removeSprite(this);
            }
        }
    }

    private boolean isBlocking(float _x, float _y, float xa, float ya) {
        int x = (int) (_x / 16);
        int y = (int) (_y / 16);
        if (x == (int) (this.x / 16) && y == (int) (this.y / 16))
            return false;

        boolean blocking = world.level.isBlocking(x, y, xa, ya);
        return blocking;
    }

    public void bumpCheck(int xTile, int yTile) {
        if (!this.alive) {
            return;
        }

        if (x + width > xTile * 16 && x - width < xTile * 16 + 16 && yTile == (int) ((y - 1) / 16)) {
            facing = -world.mario.facing;
            ya = -10;
        }
    }

    private boolean move(float xa, float ya) {
        while (xa > 8) {
            if (!move(8, 0))
                return false;
            xa -= 8;
        }
        while (xa < -8) {
            if (!move(-8, 0))
                return false;
            xa += 8;
        }
        while (ya > 8) {
            if (!move(0, 8))
                return false;
            ya -= 8;
        }
        while (ya < -8) {
            if (!move(0, -8))
                return false;
            ya += 8;
        }

        boolean collide = false;
        if (ya > 0) {
            if (isBlocking(x + xa - width, y + ya, xa, 0))
                collide = true;
            else if (isBlocking(x + xa + width, y + ya, xa, 0))
                collide = true;
            else if (isBlocking(x + xa - width, y + ya + 1, xa, ya))
                collide = true;
            else if (isBlocking(x + xa + width, y + ya + 1, xa, ya))
                collide = true;
        }
        if (ya < 0) {
            if (isBlocking(x + xa, y + ya - height, xa, ya))
                collide = true;
            else if (collide || isBlocking(x + xa - width, y + ya - height, xa, ya))
                collide = true;
            else if (collide || isBlocking(x + xa + width, y + ya - height, xa, ya))
                collide = true;
        }
        if (xa > 0) {
            if (isBlocking(x + xa + width, y + ya - height, xa, ya))
                collide = true;
            if (isBlocking(x + xa + width, y + ya - height / 2, xa, ya))
                collide = true;
            if (isBlocking(x + xa + width, y + ya, xa, ya))
                collide = true;
        }
        if (xa < 0) {
            if (isBlocking(x + xa - width, y + ya - height, xa, ya))
                collide = true;
            if (isBlocking(x + xa - width, y + ya - height / 2, xa, ya))
                collide = true;
            if (isBlocking(x + xa - width, y + ya, xa, ya))
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

    @Override
    public void update() {
        if (!this.alive) {
            return;
        }

        if (life < 9) {
            y--;
            life++;
            return;
        }
        float sideWaysSpeed = 1.75f;
        if (xa > 2) {
            facing = 1;
        }
        if (xa < -2) {
            facing = -1;
        }

        xa = facing * sideWaysSpeed;

        if (!move(xa, 0))
            facing = -facing;
        onGround = false;
        move(0, ya);

        ya *= 0.85f;
        if (onGround) {
            xa *= GROUND_INERTIA;
        } else {
            xa *= AIR_INERTIA;
        }

        if (!onGround) {
            ya += 2;
        }
    }

    @Override
    public void render(Graphics og) {
        super.render(og);

        this.graphics.render(og, (int) (this.x - this.world.cameraX), (int) (this.y - this.world.cameraY));
    }
}