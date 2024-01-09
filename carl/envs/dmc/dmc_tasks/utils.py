from __future__ import annotations

from lxml import etree  # type: ignore

from carl.utils.types import Context


def adapt_context(xml_string: bytes, context: Context) -> bytes:
    """Adapts and returns the xml_string of the model with the given context."""

    mjcf = etree.fromstring(xml_string)
    default = mjcf.find("./default/")
    if default is None:
        default = etree.Element("default")
        mjcf.addnext(default)

    # adjust damping for all joints if damping is already an attribute
    if "joint_damping" in context:
        for joint_find in mjcf.findall(".//joint[@damping]"):
            joint_damping = joint_find.get("damping")
            joint_find.set(
                "damping", str(float(joint_damping) * context["joint_damping"])
            )

    # adjust stiffness for all joints if stiffness is already an attribute
    if "joint_stiffness" in context:
        for joint_find in mjcf.findall(".//joint[@stiffness]"):
            joint_stiffness = joint_find.get("stiffness")
            joint_find.set(
                "stiffness", str(float(joint_stiffness) * context["joint_stiffness"])
            )

    # set default joint damping if default/joint is not present
    joint = mjcf.find("./default/joint")
    if joint is None:
        joint = etree.Element("joint")
        default.addnext(joint)
        if "joint_damping" in context:
            def_joint_damping = 0.1
            default_joint_damping = str(
                float(def_joint_damping) * context["joint_damping"]
            )
            joint.set("damping", default_joint_damping)
        if "joint_stiffness" in context:
            default_joint_stiffness = str(context["joint_stiffness"])
            joint.set("stiffness", default_joint_stiffness)

    # adjust friction for all geom elements with friction attribute
    if (
        "friction_tangential" in context
        and "friction_torsional" in context
        and "friction_rolling" in context
    ):
        for geom_find in mjcf.findall(".//geom[@friction]"):
            friction = geom_find.get("friction").split(" ")
            frict_str = ""
            for i, (f, d) in enumerate(
                zip(
                    friction,
                    [
                        context["friction_tangential"],
                        context["friction_torsional"],
                        context["friction_rolling"],
                    ],
                )
            ):
                frict_str += str(float(f) * d) + " "
            geom_find.set("friction", frict_str[:-1])

    # adjust density for all geom elements with density attribute
    if "geom_density" in context:
        for geom_find in mjcf.findall(".//geom[@density]"):
            geom_find.set(
                "density",
                str(float(geom_find.get("density")) * context["geom_density"]),
            )

    # create default geom if it does not exist
    geom = mjcf.find("./default/geom")
    if geom is None:
        geom = etree.Element("geom")
        default.addnext(geom)

    # set default friction
    if (
        "friction_tangential" in context
        and "friction_torsional" in context
        and "friction_rolling" in context
    ):
        if geom.get("friction") is None:
            default_friction_tangential = 1.0
            default_friction_torsional = 0.005
            default_friction_rolling = 0.0001
            geom.set(
                "friction",
                " ".join(
                    [
                        str(
                            default_friction_tangential * context["friction_tangential"]
                        ),
                        str(default_friction_torsional * context["friction_torsional"]),
                        str(default_friction_rolling * context["friction_rolling"]),
                    ]
                ),
            )

    # set default density
    if "geom_density" in context:
        geom_density = geom.get("density")
        if geom_density is None:
            geom_density = 1000
            geom.set("density", str(float(geom_density) * context["geom_density"]))

    # scale all actuators with the actuator strength factor
    if "actuator_strength" in context:
        actuators = mjcf.findall("./actuator/")
        for actuator in actuators:
            gear = actuator.get("gear")
            if gear is None:
                gear = 1
            actuator.set("gear", str(float(gear) * context["actuator_strength"]))

    # find option settings and override them if they exist, otherwise create new option
    option = mjcf.find(".//option")
    import logging

    if option is None:
        option = etree.Element("option")
        mjcf.append(option)

    if "gravity" in context:
        gravity = option.get("gravity")
        logging.info(gravity)
        if gravity is not None:
            g = gravity.split(" ")
            gravity = " ".join([g[0], g[1], str(-context["gravity"])])
        else:
            gravity = " ".join(["0", "0", f"{str(-context['gravity'])}"])
        logging.info(gravity)
        option.set("gravity", gravity)

    if "wind_x" in context and "wind_y" in context and "wind_z" in context:
        wind = option.get("wind")
        if wind is not None:
            wind = " ".join(
                [
                    str(context["wind_x"]),
                    str(context["wind_y"]),
                    str(context["wind_z"]),
                ]
            )
        else:
            wind = " ".join(
                [
                    str(context["wind_x"]),
                    str(context["wind_y"]),
                    str(context["wind_z"]),
                ]
            )
        option.set("wind", wind)
    if "timestep" in context:
        option.set("timestep", str(context["timestep"]))
    if "density" in context:
        option.set("density", str(context["density"]))
    if "viscosity" in context:
        option.set("viscosity", str(context["viscosity"]))

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string
