package engine.sprites;

import java.awt.Graphics;

import engine.core.MarioSprite;
import engine.effects.DeathEffect;
import engine.graphics.MarioImage;
import engine.helper.Assets;
import engine.helper.EventType;
import engine.helper.SpriteType;

public class BulletBill extends MarioSprite {
    private MarioImage graphics;

    public BulletBill(boolean visuals, float x, float y, int dir) {
        super(x, y, SpriteType.BULLET_BILL);
        this.width = 4;
        this.height = 12;
        this.ya = -5;
        this.facing = dir;

        if (visuals) {
            this.graphics = new MarioImage(Assets.enemies, 40);
            this.graphics.originX = 8;
            this.graphics.originY = 31;
            this.graphics.width = 16;
        }
    }

    @Override
    public MarioSprite clone() {
        BulletBill sprite = new BulletBill(false, x, y, this.facing);
        sprite.xa = this.xa;
        sprite.ya = this.ya;
        sprite.width = this.width;
        sprite.height = this.height;
        return sprite;
    }

    @Override
    public void update() {
        if (!this.alive) {
            return;
        }

        super.update();
        float sideWaysSpeed = 4f;
        xa = facing * sideWaysSpeed;
        move(xa, 0);
        if (this.graphics != null) {
            this.graphics.flipX = facing == -1;
        }
    }

    @Override
    public void render(Graphics og) {
        super.render(og);
        this.graphics.render(og, (int) (this.x - this.world.cameraX), (int) (this.y - this.world.cameraY));
    }

    public void collideCheck() {
        if (!this.alive) {
            return;
        }

        float xMarioD = world.mario.x - x;
        float yMarioD = world.mario.y - y;
        if (xMarioD > -16 && xMarioD < 16) {
            if (yMarioD > -height && yMarioD < world.mario.height) {
                if (world.mario.ya > 0 && yMarioD <= 0 && (!world.mario.onGround || !world.mario.wasOnGround)) {
                    world.mario.stomp(this);
                    if (this.graphics != null) {
                        this.world.addEffect(new DeathEffect(this.x, this.y - 7, this.graphics.flipX, 43, 0));
                    }
                    this.world.removeSprite(this);
                } else {
                    this.world.addEvent(EventType.HURT, this.type.getValue());
                    world.mario.getHurt();
                }
            }
        }
    }

    private boolean move(float xa, float ya) {
        x += xa;
        return true;
    }

    public boolean fireballCollideCheck(Fireball fireball) {
        if (!this.alive) {
            return false;
        }

        float xD = fireball.x - x;
        float yD = fireball.y - y;

        if (xD > -16 && xD < 16) {
            return yD > -height && yD < fireball.height;
        }
        return false;
    }

    public boolean shellCollideCheck(Shell shell) {
        if (!this.alive) {
            return false;
        }

        float xD = shell.x - x;
        float yD = shell.y - y;

        if (xD > -16 && xD < 16) {
            if (yD > -height && yD < shell.height) {
                if (this.graphics != null) {
                    this.world.addEffect(new DeathEffect(this.x, this.y - 7, this.graphics.flipX, 43, -1));
                }
                this.world.addEvent(EventType.SHELL_KILL, this.type.getValue());
                this.world.removeSprite(this);
                return true;
            }
        }
        return false;
    }
}
