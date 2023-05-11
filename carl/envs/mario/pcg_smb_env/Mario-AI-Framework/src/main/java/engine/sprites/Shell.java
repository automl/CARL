package engine.sprites;

import java.awt.Graphics;

import engine.core.MarioSprite;
import engine.effects.DeathEffect;
import engine.graphics.MarioImage;
import engine.helper.Assets;
import engine.helper.EventType;
import engine.helper.SpriteType;

public class Shell extends MarioSprite {
    private static final float GROUND_INERTIA = 0.89f;
    private static final float AIR_INERTIA = 0.89f;

    private int shellType = 0;
    private boolean onGround = false;

    private MarioImage graphics;

    public Shell(boolean visuals, float x, float y, int shellType, String spriteCode) {
        super(x, y, SpriteType.SHELL);

        this.width = 4;
        this.height = 12;
        this.facing = 0;
        this.ya = -5;
        this.shellType = shellType;
        this.initialCode = spriteCode;

        if (visuals) {
            this.graphics = new MarioImage(Assets.enemies, shellType * 8 + 3);
            this.graphics.originX = 8;
            this.graphics.originY = 31;
            this.graphics.width = 16;
        }
    }

    @Override
    public MarioSprite clone() {
        Shell sprite = new Shell(false, this.x, this.y, this.shellType, this.initialCode);
        sprite.xa = this.xa;
        sprite.ya = this.ya;
        sprite.width = this.width;
        sprite.height = this.height;
        sprite.facing = this.facing;
        sprite.onGround = this.onGround;
        return sprite;
    }

    @Override
    public void update() {
        if (!this.alive) return;

        super.update();

        float sideWaysSpeed = 11f;

        if (xa > 2) {
            facing = 1;
        }
        if (xa < -2) {
            facing = -1;
        }

        xa = facing * sideWaysSpeed;

        if (facing != 0) {
            world.checkShellCollide(this);
        }

        if (!move(xa, 0)) {
            facing = -facing;
        }
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

        if (this.graphics != null) {
            this.graphics.flipX = facing == -1;
        }
    }

    @Override
    public void render(Graphics og) {
        super.render(og);
        this.graphics.render(og, (int) (this.x - this.world.cameraX), (int) (this.y - this.world.cameraY));
    }

    public boolean fireballCollideCheck(Fireball fireball) {
        if (!this.alive) return false;

        float xD = fireball.x - x;
        float yD = fireball.y - y;

        if (xD > -16 && xD < 16) {
            if (yD > -height && yD < fireball.height) {
                if (facing != 0)
                    return true;

                xa = fireball.facing * 2;
                ya = -5;
                if (this.graphics != null) {
                    this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 41 + this.shellType, -5));
                }
                this.world.removeSprite(this);
                return true;
            }
        }
        return false;
    }

    public void collideCheck() {
        if (!this.alive) return;

        float xMarioD = world.mario.x - x;
        float yMarioD = world.mario.y - y;
        if (xMarioD > -16 && xMarioD < 16) {
            if (yMarioD > -height && yMarioD < world.mario.height) {
                if (world.mario.ya > 0 && yMarioD <= 0 && (!world.mario.onGround || !world.mario.wasOnGround)) {
                    world.mario.stomp(this);
                    if (facing != 0) {
                        xa = 0;
                        facing = 0;
                    } else {
                        facing = world.mario.facing;
                    }
                } else {
                    if (facing != 0) {
                        world.addEvent(EventType.HURT, this.type.getValue());
                        world.mario.getHurt();
                    } else {
                        world.addEvent(EventType.KICK, this.type.getValue());
                        world.mario.kick(this);
                        facing = world.mario.facing;
                    }
                }
            }
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

    private boolean isBlocking(float _x, float _y, float xa, float ya) {
        int x = (int) (_x / 16);
        int y = (int) (_y / 16);
        if (x == (int) (this.x / 16) && y == (int) (this.y / 16))
            return false;

        boolean blocking = world.level.isBlocking(x, y, xa, ya);

        if (blocking && ya == 0 && xa != 0) {
            world.bump(x, y, true);
        }

        return blocking;
    }

    public void bumpCheck(int xTile, int yTile) {
        if (!this.alive) return;

        if (x + width > xTile * 16 && x - width < xTile * 16 + 16 && yTile == (int) ((y - 1) / 16)) {
            facing = -world.mario.facing;
            ya = -10;
        }
    }

    public boolean shellCollideCheck(Shell shell) {
        if (!this.alive) return false;

        float xD = shell.x - x;
        float yD = shell.y - y;

        if (xD > -16 && xD < 16) {
            if (yD > -height && yD < shell.height) {
                this.world.addEvent(EventType.SHELL_KILL, this.type.getValue());
                if (this != shell) {
                    this.world.removeSprite(shell);
                }
                this.world.removeSprite(this);
                return true;
            }
        }
        return false;
    }
}