import Box2D


def safe_destroy(world: Box2D.b2World, bodies):
    for body in bodies:
        try:
            world.DestroyBody(body)
        except AssertionError as error:
            if str(error) != "m_bodyCount > 0":
                raise error
