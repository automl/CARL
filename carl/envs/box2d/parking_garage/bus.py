import numpy as np
import math
import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
    shape,
    prismaticJointDef,
    ropeJointDef,
    fixtureDef,
    distanceJointDef,
)
from gym.envs.box2d.car_dynamics import Car

__author__ = "André Biedenkapp"

"""
Original Simulator parameters from gym.envs.box2d.car_dynamics.Car
If we replace one value with some other we comment the value here and replace it below with our own values
"""
SIZE = 0.02
WHEEL_R = 27
WHEEL_COLOR = (0.0, 0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)
MUD_COLOR = (0.4, 0.4, 0.0)

"""
Changed and added Simulator parameters
"""
ENGINE_POWER = 50_000_000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 6_000 * SIZE * SIZE
FRICTION_LIMIT = (
    1_250_000 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)

MOTOR_WHEEL_COLOR = (0.6, 0.6, 0.8)
WHEEL_W = 20
# Car Polys
WHEELPOS = [(-85, +650), (+85, +650), (-85, -30), (+85, -30)]
HULL_POLY1 = [(-70, +700), (+70, +700), (+80, +550), (-80, +550)]
HULL_POLY2 = [(-90, +550), (+90, +550), (+90, -60), (-90, -60)]

# Polys for small trailer
STRAILER_POLY = [
    (-15, -70),
    (+15, -70),
    (-60, -100),
    (+60, -100),
    (-60, -240),
    (+60, -240)
    # (-15, -130), (+15, -130),
    # (-60, -160), (+60, -160),
    # (-60, -300), (+60, -300)
]
STRAILERWHEELPOS = [(-65, -170), (+65, -170)]

# Polys for large trailer
ATRAILER_POLY = [(-90, -80), (+90, -80), (-90, -110), (+90, -110)]
ATRAILER_POLY2 = [
    (-40, -80),
    (+40, -80),
    (-40, -140),
    (+40, -140),
]
ATRAILER_POLY3 = [(-90, -140), (+90, -140), (-90, -640), (+90, -640)]
ATRAILERWHEELPOS = [(-95, -95), (+95, -95), (-95, -605), (+95, -605)]


class Bus(Car):
    """
    Different body to the original OpenAI car. We also added a brake bias with 40% front and 60% rear break bias
    """

    def _init_extra_params(self):
        self.rwd = True  # Flag to determine which wheels are driven
        self.fwd = False  # Flag to determine which wheels are driven
        self.trailer_type = (
            0  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )

    def __init__(self, world, init_angle, init_x, init_y):
        self._init_extra_params()
        self.world = world

        ##### SETUP MAIN BODY ####
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY1]
                    ),
                    density=0.66,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY2]
                    ),
                    density=0.2,
                ),
            ],
        )
        self.hull.color = (1.0, 0.85, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * SIZE, y * front_k * SIZE)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            if wy > 0 and self.fwd:
                w.color = MOTOR_WHEEL_COLOR
            elif wy < 0 and self.rwd:
                w.color = MOTOR_WHEEL_COLOR
            else:
                w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.7,
                upperAngle=+0.7,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)

        ##### SETUP SMALL TRAILER ####
        if self.trailer_type == 1:
            self.trailer = self.world.CreateDynamicBody(
                angle=init_angle,
                position=(init_x, init_y),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in STRAILER_POLY]
                    ),
                    density=2.0,
                ),
            )
            self.trailer.color = (0.0, 0.0, 0.8)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=self.trailer,
                localAnchorA=(0, -60 * SIZE),
                localAnchorB=(0, -70 * SIZE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 100 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.9,
                upperAngle=+0.9,
            )
            self.trailer.joint = self.world.CreateJoint(rjd)
            for wx, wy in STRAILERWHEELPOS:
                front_k = 1.0 if wy > 0 else 1.0
                w = self.world.CreateDynamicBody(
                    position=(init_x + wx * SIZE, init_y + wy * SIZE),
                    angle=init_angle,
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=[
                                (x * front_k * SIZE, y * front_k * SIZE)
                                for x, y in WHEEL_POLY
                            ]
                        ),
                        density=0.1,
                        categoryBits=0x0020,
                        maskBits=0x001,
                        restitution=0.0,
                    ),
                )
                w.wheel_rad = front_k * WHEEL_R * SIZE
                w.color = WHEEL_COLOR
                w.gas = 0.0
                w.brake = 0.0
                w.steer = 0.0
                w.phase = 0.0  # wheel angle
                w.omega = 0.0  # angular velocity
                w.skid_start = None
                w.skid_particle = None
                rjd = revoluteJointDef(
                    bodyA=self.trailer,
                    bodyB=w,
                    localAnchorA=(wx * SIZE, wy * SIZE),
                    localAnchorB=(0, 0),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=180 * 900 * SIZE * SIZE,
                    motorSpeed=0,
                    lowerAngle=-0.4,
                    upperAngle=+0.4,
                )
                w.joint = self.world.CreateJoint(rjd)
                w.tiles = set()
                w.userData = w
                self.wheels.append(w)
            self.drawlist = self.wheels + [self.hull, self.trailer]

        ##### SETUP LARGE TRAILER ####
        elif self.trailer_type == 2:
            self.trailer_axel = self.world.CreateDynamicBody(
                angle=init_angle,
                position=(init_x, init_y),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in ATRAILER_POLY]
                    ),
                    density=0.5,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            self.trailer = self.world.CreateDynamicBody(
                angle=init_angle,
                position=(init_x, init_y),
                fixtures=[
                    fixtureDef(
                        shape=polygonShape(
                            vertices=[(x * SIZE, y * SIZE) for x, y in ATRAILER_POLY2]
                        ),
                        density=5.0,
                        categoryBits=0x0020,
                        maskBits=0x001,
                    ),
                    fixtureDef(
                        shape=polygonShape(
                            vertices=[(x * SIZE, y * SIZE) for x, y in ATRAILER_POLY3]
                        ),
                        density=1.5,
                        categoryBits=0x0020,
                        maskBits=0x001,
                    ),
                ],
            )
            rjd = distanceJointDef(
                bodyA=self.hull,
                bodyB=self.trailer_axel,
                localAnchorA=(0, -60 * SIZE),
                localAnchorB=(-7.5 * SIZE, -80 * SIZE),
                dampingRatio=0,
                frequencyHz=500,
                length=1.25,
            )
            self.trailer_axel.joint = self.world.CreateJoint(rjd)
            rjd = distanceJointDef(
                bodyA=self.hull,
                bodyB=self.trailer_axel,
                localAnchorA=(0, -60 * SIZE),
                localAnchorB=(+7.5 * SIZE, -80 * SIZE),
                dampingRatio=0,
                frequencyHz=500,
                length=1.25,
            )
            self.trailer_axel.joint = self.world.CreateJoint(rjd)
            self.trailer_axel.color = (0.0, 0.8, 0.8)
            rjd = revoluteJointDef(
                bodyA=self.trailer_axel,
                bodyB=self.trailer,
                localAnchorA=(0.0, -95 * SIZE),
                localAnchorB=(0, -95 * SIZE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=360 * 3000 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.5,
                upperAngle=+0.5,
            )
            self.trailer.color = (0.0, 0.0, 0.8)
            self.trailer.joint = self.world.CreateJoint(rjd)
            for wx, wy in ATRAILERWHEELPOS:
                front_k = 1.0 if wy > 0 else 1.0
                w = self.world.CreateDynamicBody(
                    position=(init_x + wx * SIZE, init_y + wy * SIZE),
                    angle=init_angle,
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=[
                                (x * front_k * SIZE, y * front_k * SIZE)
                                for x, y in WHEEL_POLY
                            ]
                        ),
                        density=0.1,
                        categoryBits=0x0020,
                        maskBits=0x001,
                        restitution=0.0,
                    ),
                )
                w.wheel_rad = front_k * WHEEL_R * SIZE
                w.color = WHEEL_COLOR
                w.gas = 0.0
                w.brake = 0.0
                w.steer = 0.0
                w.phase = 0.0  # wheel angle
                w.omega = 0.0  # angular velocity
                w.skid_start = None
                w.skid_particle = None
                rjd = revoluteJointDef(
                    bodyA=self.trailer if wy < -170 else self.trailer_axel,
                    bodyB=w,
                    localAnchorA=(wx * SIZE, wy * SIZE),
                    localAnchorB=(0, 0),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=180 * 900 * SIZE * SIZE,
                    motorSpeed=0,
                    lowerAngle=-0.4,
                    upperAngle=+0.4,
                )
                w.joint = self.world.CreateJoint(rjd)
                w.tiles = set()
                w.userData = w
                self.wheels.append(w)
            self.drawlist = self.wheels + [self.hull, self.trailer, self.trailer_axel]
        else:
            self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        if self.fwd:
            for w in self.wheels[:2]:
                diff = gas - w.gas
                if diff > 0.1:
                    diff = 0.1  # gradually increase, but stop immediately
                w.gas += diff
        if self.rwd:
            for w in self.wheels[2:4]:
                diff = gas - w.gas
                if diff > 0.1:
                    diff = 0.1  # gradually increase, but stop immediately
                w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        for w in self.wheels[:2]:
            w.brake = b * 0.4
        for w in self.wheels[2:4]:
            w.brake = b * 0.6
        if self.trailer_type == 1:
            for w in self.wheels[4:6]:
                w.brake = b * 0.7
        if self.trailer_type == 2:
            for w in self.wheels[4:]:
                w.brake = b * 0.8

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        """
        Copy of the original step function of 'gym.envs.box2d.car_dynamics.Car' needed to accept different
        Engin powers or other fixed parameters
        :param dt:
        :return:
        """
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * ENGINE_POWER
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )


class FWDBus(Bus):
    """
    Front wheel driven race car
    """

    def _init_extra_params(self):
        self.rwd = False  # Flag to determine which wheels are driven
        self.fwd = True  # Flag to determine which wheels are driven
        self.trailer_type = (
            0  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class AWDBus(Bus):
    """
    4x4 wheel driven race car
    """

    def _init_extra_params(self):
        self.rwd = True  # Flag to determine which wheels are driven
        self.fwd = True  # Flag to determine which wheels are driven
        self.trailer_type = (
            0  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class BusSmallTrailer(Bus):
    """
    Bus with small trailer attached
    """

    def _init_extra_params(self):
        self.rwd = True  # Flag to determine which wheels are driven
        self.fwd = False  # Flag to determine which wheels are driven
        self.trailer_type = (
            1  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class FWDBusSmallTrailer(Bus):
    """
    Front wheel driven race car
    """

    def _init_extra_params(self):
        self.rwd = False  # Flag to determine which wheels are driven
        self.fwd = True  # Flag to determine which wheels are driven
        self.trailer_type = (
            1  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class AWDBusSmallTrailer(Bus):
    """
    4x4 wheel driven race car
    """

    def _init_extra_params(self):
        self.rwd = True  # Flag to determine which wheels are driven
        self.fwd = True  # Flag to determine which wheels are driven
        self.trailer_type = (
            1  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class BusLargeTrailer(Bus):
    """
    Bus with small trailer attached
    """

    def _init_extra_params(self):
        self.rwd = True  # Flag to determine which wheels are driven
        self.fwd = False  # Flag to determine which wheels are driven
        self.trailer_type = (
            2  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class FWDBusLargeTrailer(Bus):
    """
    Front wheel driven race car
    """

    def _init_extra_params(self):
        self.rwd = False  # Flag to determine which wheels are driven
        self.fwd = True  # Flag to determine which wheels are driven
        self.trailer_type = (
            2  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )


class AWDBusLargeTrailer(Bus):
    """
    4x4 wheel driven race car
    """

    def _init_extra_params(self):
        self.rwd = True  # Flag to determine which wheels are driven
        self.fwd = True  # Flag to determine which wheels are driven
        self.trailer_type = (
            2  # Determines which trailer to attach 0 -> none, 1 -> small, 2 -> large
        )
