/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package agents.spencerSchumann;

/**
 * @author Spencer
 */
public class Enemy {

    public int type;
    public float x;
    public float y;
    public float vx;
    public float height;
    public float width;

    public boolean winged;
    public boolean safeTop;

    public static final int KIND_NONE = 0;
    public static final int KIND_GOOMBA = 2;
    public static final int KIND_GOOMBA_WINGED = 3;
    public static final int KIND_RED_KOOPA = 4;
    public static final int KIND_RED_KOOPA_WINGED = 5;
    public static final int KIND_GREEN_KOOPA = 6;
    public static final int KIND_GREEN_KOOPA_WINGED = 7;
    public static final int KIND_BULLET_BILL = 10;
    public static final int KIND_SPIKEY = 8;
    public static final int KIND_SPIKEY_WINGED = 9;
    public static final int KIND_ENEMY_FLOWER = 11;
    public static final int KIND_SHELL = 14;
    public static final int KIND_MUSHROOM = 12;
    public static final int KIND_FIRE_FLOWER = 13;
    public static final int KIND_FIREBALL = 16;

    public Enemy(int type, float x, float y) {
        this.type = type;
        this.x = x;
        this.y = y;

        height = 24.0f;
        width = 16.0f;
        if (type == KIND_BULLET_BILL) {
            height = 12.0f;
            width = 24.0f;
        }

        switch (type) {
            case KIND_GOOMBA_WINGED:
            case KIND_RED_KOOPA_WINGED:
            case KIND_GREEN_KOOPA_WINGED:
            case KIND_SPIKEY_WINGED:
                winged = true;
                break;
            default:
                break;
        }

        switch (type) {
            case KIND_SPIKEY:
            case KIND_SPIKEY_WINGED:
            case KIND_ENEMY_FLOWER:
                safeTop = false;
                break;
            default:
                safeTop = true;
                break;
        }
    }

    @Override
    public Enemy clone() {
        Enemy e = new Enemy(type, x, y);
        e.vx = vx;
        return e;
    }
}
