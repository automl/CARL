package engine.sprites;

import java.awt.Graphics;

import engine.core.MarioSprite;
import engine.effects.DeathEffect;
import engine.effects.SquishEffect;
import engine.graphics.MarioImage;
import engine.helper.Assets;
import engine.helper.EventType;
import engine.helper.SpriteType;

public class Enemy extends MarioSprite {
    private static final float GROUND_INERTIA = 0.89f;
    private static final float AIR_INERTIA = 0.89f;

    protected boolean onGround = false;
    protected boolean avoidCliffs = true;
    protected boolean winged = true;
    protected boolean noFireballDeath;

    protected float runTime;
    protected int wingTime = 0;
    protected MarioImage wingGraphics;
    protected MarioImage graphics;

    public Enemy(boolean visuals, float x, float y, int dir, SpriteType type) {
        super(x, y, type);
        this.width = 4;
        this.height = 24;
        if (this.type != SpriteType.RED_KOOPA && this.type != SpriteType.GREEN_KOOPA
                && this.type != SpriteType.RED_KOOPA_WINGED && this.type != SpriteType.GREEN_KOOPA_WINGED) {
            this.height = 12;
        }
        this.winged = this.type.getValue() % 2 == 1;
        this.avoidCliffs = this.type == SpriteType.RED_KOOPA || this.type == SpriteType.RED_KOOPA_WINGED;
        this.noFireballDeath = this.type == SpriteType.SPIKY || this.type == SpriteType.SPIKY_WINGED;
        this.facing = dir;
        if (this.facing == 0) {
            this.facing = 1;
        }

        if (visuals) {
            this.graphics = new MarioImage(Assets.enemies, this.type.getStartIndex());
            this.graphics.originX = 8;
            this.graphics.originY = 31;
            this.graphics.width = 16;

            this.wingGraphics = new MarioImage(Assets.enemies, 32);
            this.wingGraphics.originX = 16;
            this.wingGraphics.originY = 31;
            this.wingGraphics.width = 16;
        }
    }

    @Override
    public MarioSprite clone() {
        Enemy e = new Enemy(false, this.x, this.y, this.facing, this.type);
        e.xa = this.xa;
        e.ya = this.ya;
        e.initialCode = this.initialCode;
        e.width = this.width;
        e.height = this.height;
        e.onGround = this.onGround;
        e.winged = this.winged;
        e.avoidCliffs = this.avoidCliffs;
        e.noFireballDeath = this.noFireballDeath;
        return e;
    }

    public void collideCheck() {
        if (!this.alive) {
            return;
        }

        float xMarioD = world.mario.x - x;
        float yMarioD = world.mario.y - y;
        if (xMarioD > -width * 2 - 4 && xMarioD < width * 2 + 4) {
            if (yMarioD > -height && yMarioD < world.mario.height) {
                if (type != SpriteType.SPIKY && type != SpriteType.SPIKY_WINGED && type != SpriteType.ENEMY_FLOWER &&
                        world.mario.ya > 0 && yMarioD <= 0 && (!world.mario.onGround || !world.mario.wasOnGround)) {
                    world.mario.stomp(this);
                    if (winged) {
                        winged = false;
                        ya = 0;
                    } else {
                        if (type == SpriteType.GREEN_KOOPA || type == SpriteType.GREEN_KOOPA_WINGED) {
                            this.world.addSprite(new Shell(this.graphics != null, x, y, 1, this.initialCode));
                        } else if (type == SpriteType.RED_KOOPA || type == SpriteType.RED_KOOPA_WINGED) {
                            this.world.addSprite(new Shell(this.graphics != null, x, y, 0, this.initialCode));
                        } else if (type == SpriteType.GOOMBA || type == SpriteType.GOOMBA_WINGED) {
                            if (this.graphics != null) {
                                this.world.addEffect(new SquishEffect(this.x, this.y - 7));
                            }
                        }
                        this.world.addEvent(EventType.STOMP_KILL, this.type.getValue());
                        this.world.removeSprite(this);
                    }
                } else {
                    this.world.addEvent(EventType.HURT, this.type.getValue());
                    world.mario.getHurt();
                }
            }
        }
    }

    private void updateGraphics() {
        wingTime++;
        this.wingGraphics.index = 32 + wingTime / 4 % 2;

        this.graphics.flipX = this.facing == -1;
        runTime += (Math.abs(xa)) + 5;

        int runFrame = ((int) (runTime / 20)) % 2;

        if (!onGround) {
            runFrame = 1;
        }
        if (winged)
            runFrame = wingTime / 4 % 2;

        this.graphics.index = this.type.getStartIndex() + runFrame;
    }

    @Override
    public void update() {
        if (!this.alive) {
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

        ya *= winged ? 0.95f : 0.85f;
        if (onGround) {
            xa *= GROUND_INERTIA;
        } else {
            xa *= AIR_INERTIA;
        }

        if (!onGround) {
            if (winged) {
                ya += 0.6f;
            } else {
                ya += 2;
            }
        } else if (winged) {
            ya = -10;
        }

        if (this.graphics != null) {
            this.updateGraphics();
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

            if (avoidCliffs && onGround
                    && !world.level.isBlocking((int) ((x + xa + width) / 16), (int) ((y) / 16 + 1), xa, 1))
                collide = true;
        }
        if (xa < 0) {
            if (isBlocking(x + xa - width, y + ya - height, xa, ya))
                collide = true;
            if (isBlocking(x + xa - width, y + ya - height / 2, xa, ya))
                collide = true;
            if (isBlocking(x + xa - width, y + ya, xa, ya))
                collide = true;

            if (avoidCliffs && onGround
                    && !world.level.isBlocking((int) ((x + xa - width) / 16), (int) ((y) / 16 + 1), xa, 1))
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

        return blocking;
    }

    public boolean shellCollideCheck(Shell shell) {
        if (!this.alive) {
            return false;
        }

        float xD = shell.x - x;
        float yD = shell.y - y;

        if (xD > -16 && xD < 16) {
            if (yD > -height && yD < shell.height) {
                xa = shell.facing * 2;
                ya = -5;
                this.world.addEvent(EventType.SHELL_KILL, this.type.getValue());
                if (this.graphics != null) {
                    if (this.type == SpriteType.GREEN_KOOPA || this.type == SpriteType.GREEN_KOOPA_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 42, -5));
                    } else if (this.type == SpriteType.RED_KOOPA || this.type == SpriteType.RED_KOOPA_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 41, -5));
                    } else if (this.type == SpriteType.GOOMBA || this.type == SpriteType.GOOMBA_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 44, -5));
                    } else if (this.type == SpriteType.SPIKY || this.type == SpriteType.SPIKY_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 45, -5));
                    }
                }
                this.world.removeSprite(this);
                return true;
            }
        }
        return false;
    }

    public boolean fireballCollideCheck(Fireball fireball) {
        if (!this.alive) {
            return false;
        }

        float xD = fireball.x - x;
        float yD = fireball.y - y;

        if (xD > -16 && xD < 16) {
            if (yD > -height && yD < fireball.height) {
                if (noFireballDeath)
                    return true;

                xa = fireball.facing * 2;
                ya = -5;
                this.world.addEvent(EventType.FIRE_KILL, this.type.getValue());
                if (this.graphics != null) {
                    if (this.type == SpriteType.GREEN_KOOPA || this.type == SpriteType.GREEN_KOOPA_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 42, -5));
                    } else if (this.type == SpriteType.RED_KOOPA || this.type == SpriteType.RED_KOOPA_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 41, -5));
                    } else if (this.type == SpriteType.GOOMBA || this.type == SpriteType.GOOMBA_WINGED) {
                        this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 44, -5));
                    }
                }
                this.world.removeSprite(this);
                return true;
            }
        }
        return false;
    }

    public void bumpCheck(int xTile, int yTile) {
        if (!this.alive) {
            return;
        }

        if (x + width > xTile * 16 && x - width < xTile * 16 + 16 && yTile == (int) ((y - 1) / 16)) {
            xa = -world.mario.facing * 2;
            ya = -5;
            if (this.graphics != null) {
                if (this.type == SpriteType.GREEN_KOOPA || this.type == SpriteType.GREEN_KOOPA_WINGED) {
                    this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 42, -5));
                } else if (this.type == SpriteType.RED_KOOPA || this.type == SpriteType.RED_KOOPA_WINGED) {
                    this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 41, -5));
                } else if (this.type == SpriteType.GOOMBA || this.type == SpriteType.GOOMBA_WINGED) {
                    this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 44, -5));
                } else if (this.type == SpriteType.SPIKY || this.type == SpriteType.SPIKY_WINGED) {
                    this.world.addEffect(new DeathEffect(this.x, this.y, this.graphics.flipX, 45, -5));
                }
            }
            this.world.removeSprite(this);
        }
    }

    @Override
    public void render(Graphics og) {
        if (winged) {
            if (type != SpriteType.RED_KOOPA && type != SpriteType.GREEN_KOOPA && type != SpriteType.RED_KOOPA_WINGED
                    && type != SpriteType.GREEN_KOOPA_WINGED) {
                this.wingGraphics.flipX = false;
                this.wingGraphics.render(og, (int) (this.x - this.world.cameraX - 6), (int) (this.y - this.world.cameraY - 6));
                this.wingGraphics.flipX = true;
                this.wingGraphics.render(og, (int) (this.x - this.world.cameraX + 22), (int) (this.y - this.world.cameraY - 6));
            }
        }

        this.graphics.render(og, (int) (this.x - this.world.cameraX), (int) (this.y - this.world.cameraY));

        if (winged) {
            if (type == SpriteType.RED_KOOPA || type == SpriteType.GREEN_KOOPA || type == SpriteType.RED_KOOPA_WINGED
                    || type == SpriteType.GREEN_KOOPA_WINGED) {
                int shiftX = -1;
                if (this.graphics.flipX) {
                    shiftX = 17;
                }
                this.wingGraphics.flipX = this.graphics.flipX;
                this.wingGraphics.render(og, (int) (this.x - this.world.cameraX + shiftX), (int) (this.y - this.world.cameraY - 8));
            }
        }
    }

}
