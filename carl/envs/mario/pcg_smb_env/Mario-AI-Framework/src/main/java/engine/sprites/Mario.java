package engine.sprites;

import java.awt.Graphics;

import engine.core.MarioSprite;
import engine.graphics.MarioImage;
import engine.helper.Assets;
import engine.helper.EventType;
import engine.helper.MarioActions;
import engine.helper.SpriteType;
import engine.helper.TileFeature;

public class Mario extends MarioSprite {
    public boolean isLarge, isFire;
    public boolean onGround, wasOnGround, isDucking, canShoot, mayJump;
    public boolean[] actions = null;
    public int jumpTime = 0;

    private float xJumpSpeed, yJumpSpeed = 0;
    private int invulnerableTime = 0;

    private float marioFrameSpeed = 0;
    private boolean oldLarge, oldFire = false;
    private MarioImage graphics = null;

    // stats
    private float xJumpStart = -100;

    private final float GROUND_INERTIA = 0.89f;
    private final float AIR_INERTIA = 0.89f;
    private final int POWERUP_TIME = 3;

    public Mario(boolean visuals, float x, float y) {
        super(x + 8, y + 15, SpriteType.MARIO);
        this.isLarge = this.oldLarge = false;
        this.isFire = this.oldFire = false;
        this.width = 4;
        this.height = 24;
        if (visuals) {
            graphics = new MarioImage(Assets.smallMario, 0);
        }
    }

    @Override
    public MarioSprite clone() {
        Mario sprite = new Mario(false, x - 8, y - 15);
        sprite.xa = this.xa;
        sprite.ya = this.ya;
        sprite.initialCode = this.initialCode;
        sprite.width = this.width;
        sprite.height = this.height;
        sprite.facing = this.facing;
        sprite.isLarge = isLarge;
        sprite.isFire = isFire;
        sprite.wasOnGround = wasOnGround;
        sprite.onGround = onGround;
        sprite.isDucking = isDucking;
        sprite.canShoot = canShoot;
        sprite.mayJump = mayJump;
        sprite.actions = new boolean[this.actions.length];
        for (int i = 0; i < this.actions.length; i++) {
            sprite.actions[i] = this.actions[i];
        }
        sprite.xJumpSpeed = xJumpSpeed;
        sprite.yJumpSpeed = yJumpSpeed;
        sprite.invulnerableTime = invulnerableTime;
        sprite.jumpTime = jumpTime;
        sprite.xJumpStart = xJumpStart;
        return sprite;
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
                jumpTime = 0;
                this.ya = 0;
            }
            if (ya > 0) {
                y = (int) ((y - 1) / 16 + 1) * 16 - 1;
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
        int xTile = (int) (_x / 16);
        int yTile = (int) (_y / 16);
        if (xTile == (int) (this.x / 16) && yTile == (int) (this.y / 16))
            return false;

        boolean blocking = world.level.isBlocking(xTile, yTile, xa, ya);
        int block = world.level.getBlock(xTile, yTile);

        if (TileFeature.getTileType(block).contains(TileFeature.PICKABLE)) {
            this.world.addEvent(EventType.COLLECT, block);
            this.collectCoin();
            world.level.setBlock(xTile, yTile, 0);
        }
        if (blocking && ya < 0) {
            world.bump(xTile, yTile, isLarge);
        }
        return blocking;
    }

    public void updateGraphics() {
        if (!this.alive) {
            return;
        }

        boolean currentLarge, currentFire;
        if (this.world.pauseTimer > 0) {
            currentLarge = (this.world.pauseTimer / 2) % 2 == 0 ? this.oldLarge : this.isLarge;
            currentFire = (this.world.pauseTimer / 2) % 2 == 0 ? this.oldFire : this.isFire;
        } else {
            currentLarge = this.isLarge;
            currentFire = this.isFire;
        }
        if (currentLarge) {
            this.graphics.sheet = Assets.mario;
            if (currentFire) {
                this.graphics.sheet = Assets.fireMario;
            }

            graphics.originX = 16;
            graphics.originY = 31;
            graphics.width = graphics.height = 32;
        } else {
            this.graphics.sheet = Assets.smallMario;
            graphics.originX = 8;
            graphics.originY = 15;
            graphics.width = graphics.height = 16;
        }

        this.marioFrameSpeed += Math.abs(xa) + 5;
        if (Math.abs(xa) < 0.5f) {
            this.marioFrameSpeed = 0;
        }

        graphics.visible = ((invulnerableTime / 2) & 1) == 0;
        graphics.flipX = facing == -1;

        int frameIndex = 0;
        if (currentLarge) {
            frameIndex = ((int) (marioFrameSpeed / 20)) % 4;
            if (frameIndex == 3)
                frameIndex = 1;
            if (Math.abs(xa) > 10)
                frameIndex += 3;
            if (!onGround) {
                if (Math.abs(xa) > 10)
                    frameIndex = 6;
                else
                    frameIndex = 5;
            }
        } else {
            frameIndex = ((int) (marioFrameSpeed / 20)) % 2;
            if (Math.abs(xa) > 10)
                frameIndex += 2;
            if (!onGround) {
                if (Math.abs(xa) > 10)
                    frameIndex = 5;
                else
                    frameIndex = 4;
            }
        }

        if (onGround && ((facing == -1 && xa > 0) || (facing == 1 && xa < 0))) {
            if (xa > 1 || xa < -1)
                frameIndex = currentLarge ? 8 : 7;
        }

        if (currentLarge && isDucking) {
            frameIndex = 13;
        }

        graphics.index = frameIndex;
    }

    @Override
    public void update() {
        if (!this.alive) {
            return;
        }

        if (invulnerableTime > 0) {
            invulnerableTime--;
        }
        this.wasOnGround = this.onGround;

        float sideWaysSpeed = actions[MarioActions.SPEED.getValue()] ? 1.2f : 0.6f;

        if (onGround) {
            isDucking = actions[MarioActions.DOWN.getValue()] && isLarge;
        }

        if (isLarge) {
            height = isDucking ? 12 : 24;
        } else {
            height = 12;
        }

        if (xa > 2) {
            facing = 1;
        }
        if (xa < -2) {
            facing = -1;
        }

        if (actions[MarioActions.JUMP.getValue()] || (jumpTime < 0 && !onGround)) {
            if (jumpTime < 0) {
                xa = xJumpSpeed;
                ya = -jumpTime * yJumpSpeed;
                jumpTime++;
            } else if (onGround && mayJump) {
                xJumpSpeed = 0;
                yJumpSpeed = -1.9f;
                jumpTime = 7;
                ya = jumpTime * yJumpSpeed;
                onGround = false;
                if (!(isBlocking(x, y - 4 - height, 0, -4) || isBlocking(x - width, y - 4 - height, 0, -4)
                        || isBlocking(x + width, y - 4 - height, 0, -4))) {
                    this.xJumpStart = this.x;
                    this.world.addEvent(EventType.JUMP, 0);
                }
            } else if (jumpTime > 0) {
                xa += xJumpSpeed;
                ya = jumpTime * yJumpSpeed;
                jumpTime--;
            }
        } else {
            jumpTime = 0;
        }

        if (actions[MarioActions.LEFT.getValue()] && !isDucking) {
            xa -= sideWaysSpeed;
            if (jumpTime >= 0)
                facing = -1;
        }

        if (actions[MarioActions.RIGHT.getValue()] && !isDucking) {
            xa += sideWaysSpeed;
            if (jumpTime >= 0)
                facing = 1;
        }

        if (actions[MarioActions.SPEED.getValue()] && canShoot && isFire && world.fireballsOnScreen < 2) {
            world.addSprite(new Fireball(this.graphics != null, x + facing * 6, y - 20, facing));
        }

        canShoot = !actions[MarioActions.SPEED.getValue()];

        mayJump = onGround && !actions[MarioActions.JUMP.getValue()];

        if (Math.abs(xa) < 0.5f) {
            xa = 0;
        }

        onGround = false;
        move(xa, 0);
        move(0, ya);
        if (!wasOnGround && onGround && this.xJumpStart >= 0) {
            this.world.addEvent(EventType.LAND, 0);
            this.xJumpStart = -100;
        }

        if (x < 0) {
            x = 0;
            xa = 0;
        }

        if (x > world.level.exitTileX * 16) {
            x = world.level.exitTileX * 16;
            xa = 0;
            this.world.win();
        }

        ya *= 0.85f;
        if (onGround) {
            xa *= GROUND_INERTIA;
        } else {
            xa *= AIR_INERTIA;
        }

        if (!onGround) {
            ya += 3;
        }

        if (this.graphics != null) {
            this.updateGraphics();
        }
    }

    public void stomp(Enemy enemy) {
        if (!this.alive) {
            return;
        }
        float targetY = enemy.y - enemy.height / 2;
        move(0, targetY - y);

        xJumpSpeed = 0;
        yJumpSpeed = -1.9f;
        jumpTime = 8;
        ya = jumpTime * yJumpSpeed;
        onGround = false;
        invulnerableTime = 1;
    }

    public void stomp(Shell shell) {
        if (!this.alive) {
            return;
        }
        float targetY = shell.y - shell.height / 2;
        move(0, targetY - y);

        xJumpSpeed = 0;
        yJumpSpeed = -1.9f;
        jumpTime = 8;
        ya = jumpTime * yJumpSpeed;
        onGround = false;
        invulnerableTime = 1;
    }

    public void getHurt() {
        if (invulnerableTime > 0 || !this.alive)
            return;

        if (isLarge) {
            world.pauseTimer = 3 * POWERUP_TIME;
            this.oldLarge = this.isLarge;
            this.oldFire = this.isFire;
            if (isFire) {
                this.isFire = false;
            } else {
                this.isLarge = false;
            }
            invulnerableTime = 32;
        } else {
            if (this.world != null) {
                this.world.lose();
            }
        }
    }

    public void getFlower() {
        if (!this.alive) {
            return;
        }

        if (!isFire) {
            world.pauseTimer = 3 * POWERUP_TIME;
            this.oldFire = this.isFire;
            this.oldLarge = this.isLarge;
            this.isFire = true;
            this.isLarge = true;
        } else {
            this.collectCoin();
        }
    }

    public void getMushroom() {
        if (!this.alive) {
            return;
        }

        if (!isLarge) {
            world.pauseTimer = 3 * POWERUP_TIME;
            this.oldFire = this.isFire;
            this.oldLarge = this.isLarge;
            this.isLarge = true;
        } else {
            this.collectCoin();
        }
    }

    public void kick(Shell shell) {
        if (!this.alive) {
            return;
        }

        invulnerableTime = 1;
    }

    public void stomp(BulletBill bill) {
        if (!this.alive) {
            return;
        }

        float targetY = bill.y - bill.height / 2;
        move(0, targetY - y);

        xJumpSpeed = 0;
        yJumpSpeed = -1.9f;
        jumpTime = 8;
        ya = jumpTime * yJumpSpeed;
        onGround = false;
        invulnerableTime = 1;
    }

    public String getMarioType() {
        if (isFire) {
            return "fire";
        }
        if (isLarge) {
            return "large";
        }
        return "small";
    }

    public void collect1Up() {
        if (!this.alive) {
            return;
        }

        this.world.lives++;
    }

    public void collectCoin() {
        if (!this.alive) {
            return;
        }

        this.world.coins++;
        if (this.world.coins % 100 == 0) {
            collect1Up();
        }
    }

    @Override
    public void render(Graphics og) {
        super.render(og);

        this.graphics.render(og, (int) (this.x - this.world.cameraX), (int) (this.y - this.world.cameraY));
    }
}
