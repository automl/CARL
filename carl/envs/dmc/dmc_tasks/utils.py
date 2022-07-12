from typing import List

from lxml import etree  # type: ignore

from carl.utils.types import Context


def adapt_context(
    xml_string: bytes, context: Context, context_mask: List = []
) -> bytes:
    """Adapts and returns the xml_string of the model with the given context."""
    mjcf = etree.fromstring(xml_string)
    default = mjcf.find("./default/")
    if default is None:
        default = etree.Element("default")
        mjcf.addnext(default)

    if "joint_damping" not in context_mask:
        # adjust damping for all joints if damping is already an attribute
        for joint_find in mjcf.findall(".//joint[@damping]"):
            joint_damping = joint_find.get("damping")
            joint_find.set(
                "damping", str(float(joint_damping) * context["joint_damping"])
            )

    if "joint_stiffness" not in context_mask:
        # adjust stiffness for all joints if stiffness is already an attribute
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
        if "joint_damping" not in context_mask:
            def_joint_damping = 0.1
            default_joint_damping = str(
                float(def_joint_damping) * context["joint_damping"]
            )
            joint.set("damping", default_joint_damping)
        if "joint_stiffness" not in context_mask:
            default_joint_stiffness = str(context["joint_stiffness"])
            joint.set("stiffness", default_joint_stiffness)

    # adjust friction for all geom elements with friction attribute
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
            if (
                (i == 0 and "friction_tangential" not in context_mask)
                or (i == 1 and "friction_torsional" not in context_mask)
                or (i == 2 and "friction_rolling" not in context_mask)
            ):
                frict_str += str(float(f) * d) + " "
            else:
                frict_str += str(f) + " "
        geom_find.set("friction", frict_str[:-1])

    if "geom_density" not in context_mask:
        # adjust density for all geom elements with density attribute
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
    if geom.get("friction") is None:
        default_friction_tangential = 1.0
        default_friction_torsional = 0.005
        default_friction_rolling = 0.0001
        geom.set(
            "friction",
            " ".join(
                [
                    (
                        str(
                            default_friction_tangential * context["friction_tangential"]
                        )
                        if "friction_tangential" not in context_mask
                        else str(default_friction_tangential)
                    ),
                    (
                        str(default_friction_torsional * context["friction_torsional"])
                        if "friction_torsional" not in context_mask
                        else str(default_friction_torsional)
                    ),
                    (
                        str(default_friction_rolling * context["friction_rolling"])
                        if "friction_rolling" not in context_mask
                        else str(default_friction_rolling)
                    ),
                ]
            ),
        )

    if "geom_density" not in context_mask:
        # set default density
        geom_density = geom.get("density")
        if geom_density is None:
            geom_density = 1000
            geom.set("density", str(float(geom_density) * context["geom_density"]))

    if "actuator_strength" not in context_mask:
        # scale all actuators with the actuator strength factor
        actuators = mjcf.findall("./actuator/")
        for actuator in actuators:
            gear = actuator.get("gear")
            if gear is None:
                gear = 1
            actuator.set("gear", str(float(gear) * context["actuator_strength"]))

    # find option settings and override them if they exist, otherwise create new option
    option = mjcf.find(".//option")
    if option is None:
        option = etree.Element("option")
        mjcf.append(option)

    if "gravity" not in context_mask:
        gravity = option.get("gravity")
        if gravity is not None:
            g = gravity.split(" ")
            gravity = " ".join([g[0], g[1], str(context["gravity"])])
        else:
            gravity = " ".join(["0", "0", str(context["gravity"])])
        option.set("gravity", gravity)

    if "wind" not in context_mask:
        wind = option.get("wind")
        if wind is not None:
            w = wind.split(" ")
            wind = " ".join(
                [
                    (str(context["wind_x"]) if "wind_x" not in context_mask else w[0]),
                    (str(context["wind_y"]) if "wind_y" not in context_mask else w[1]),
                    (str(context["wind_z"]) if "wind_z" not in context_mask else w[2]),
                ]
            )
        else:
            wind = " ".join(
                [
                    (str(context["wind_x"]) if "wind_x" not in context_mask else "0"),
                    (str(context["wind_y"]) if "wind_y" not in context_mask else "0"),
                    (str(context["wind_z"]) if "wind_z" not in context_mask else "0"),
                ]
            )
        option.set("wind", wind)

    if "timestep" not in context_mask:
        option.set("timestep", str(context["timestep"]))

    if "density" not in context_mask:
        option.set("density", str(context["density"]))

    if "viscosity" not in context_mask:
        option.set("viscosity", str(context["viscosity"]))

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string
