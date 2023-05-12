package engine.helper;

import engine.core.MarioSprite;
import engine.sprites.Enemy;
import engine.sprites.FlowerEnemy;

public enum SpriteType {
    //Generic values
    NONE(0),
    UNDEF(-42),
    MARIO(-31),
    FIREBALL(16),
    GOOMBA(2, 16),
    GOOMBA_WINGED(3, 16),
    RED_KOOPA(4, 0),
    RED_KOOPA_WINGED(5, 0),
    GREEN_KOOPA(6, 8),
    GREEN_KOOPA_WINGED(7, 8),
    SPIKY(8, 24),
    SPIKY_WINGED(9, 24),
    BULLET_BILL(10, 40),
    ENEMY_FLOWER(11, 48),
    MUSHROOM(12),
    FIRE_FLOWER(13),
    SHELL(14),
    LIFE_MUSHROOM(15);

    private int value;
    private int startIndex;

    SpriteType(int newValue) {
        value = newValue;
    }

    SpriteType(int newValue, int newIndex) {
        value = newValue;
        startIndex = newIndex;
    }

    public int getValue() {
        return value;
    }

    public int getStartIndex() {
        return startIndex;
    }

    public MarioSprite spawnSprite(boolean visuals, int xTile, int yTile, int dir) {
        if (this == SpriteType.ENEMY_FLOWER) {
            return new FlowerEnemy(visuals, xTile * 16 + 17, yTile * 16 + 18);
        }
        return new Enemy(visuals, xTile * 16 + 8, yTile * 16 + 15, dir, this);
    }
}
