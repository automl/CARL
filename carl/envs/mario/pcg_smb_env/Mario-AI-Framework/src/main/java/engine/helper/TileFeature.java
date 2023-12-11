package engine.helper;

import java.util.ArrayList;

public enum TileFeature {
    BLOCK_UPPER,
    BLOCK_ALL,
    BLOCK_LOWER,
    SPECIAL,
    LIFE,
    BUMPABLE,
    BREAKABLE,
    PICKABLE,
    ANIMATED,
    SPAWNER;

    public static ArrayList<TileFeature> getTileType(int index) {
        ArrayList<TileFeature> features = new ArrayList<>();
        switch (index) {
            case 1:
            case 2:
            case 14:
            case 18:
            case 19:
            case 20:
            case 21:
            case 4:
            case 5:
            case 52:
            case 53:
                features.add(TileFeature.BLOCK_ALL);
                break;
            case 43:
            case 44:
            case 45:
            case 46:
                features.add(TileFeature.BLOCK_LOWER);
                break;
            case 48:
                features.add(TileFeature.BLOCK_UPPER);
                features.add(TileFeature.LIFE);
                features.add(TileFeature.BUMPABLE);
                break;
            case 49:
                features.add(TileFeature.BUMPABLE);
                features.add(TileFeature.BLOCK_UPPER);
                break;
            case 3:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.SPAWNER);
                break;
            case 8:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.SPECIAL);
                features.add(TileFeature.BUMPABLE);
                features.add(TileFeature.ANIMATED);
                break;
            case 11:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.BUMPABLE);
                features.add(TileFeature.ANIMATED);
                break;
            case 6:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.BREAKABLE);
                break;
            case 7:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.BUMPABLE);
                break;
            case 15:
                features.add(TileFeature.PICKABLE);
                features.add(TileFeature.ANIMATED);
                break;
            case 50:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.SPECIAL);
                features.add(TileFeature.BUMPABLE);
                break;
            case 51:
                features.add(TileFeature.BLOCK_ALL);
                features.add(TileFeature.LIFE);
                features.add(TileFeature.BUMPABLE);
                break;
        }
        return features;
    }
}
