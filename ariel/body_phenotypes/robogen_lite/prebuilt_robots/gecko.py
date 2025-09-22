from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
import numpy as np

def gecko():
    """
    Create and attach bodies/sites, then print relative orientations between
    abdomen and spine bodies to debug hinge placement/orientation.
    """

    core = CoreModule(
        index=0,
    )
    neck = HingeModule(
        index=1,
    )
    # neck.rotate(-90)
    abdomen = BrickModule(
        index=2,
    )
    spine = HingeModule(
        index=3,
    )
    # spine.rotate(90)
    butt = BrickModule(
        index=4,
    )
    
    fl_leg = HingeModule(
        index=5,
    )
    fl_leg.rotate(90)
    fl_leg2 = HingeModule(
        index=15,
    )
    fl_leg2.rotate(90)
    
    fl_flipper = BrickModule(
        index=6,
    )
    
    fr_leg = HingeModule(
        index=7,
    )
    fr_leg.rotate(-90)
    fr_leg2 = HingeModule(
        index=17,
    )
    fr_leg2.rotate(90)
    
    fr_flipper = BrickModule(
        index=8,
    )
    bl_leg = HingeModule(
        index=9,
    )
    bl_leg.rotate(45)
    bl_flipper = BrickModule(
        index=10,
    )
    br_leg = HingeModule(
        index=11,
    )
    br_leg.rotate(-45)
    br_flipper = BrickModule(
        index=12,
    )

    # Attach bodies
    core.sites[ModuleFaces.FRONT].attach_body(
        body=neck.body,
        prefix="neck",
    )

    neck.sites[ModuleFaces.FRONT].attach_body(
        body=abdomen.body,
        prefix="abdomen",
    )
    abdomen.sites[ModuleFaces.FRONT].attach_body(
        body=spine.body,
        prefix="spine",
    )
    spine.sites[ModuleFaces.FRONT].attach_body(
        body=butt.body,
        prefix="butt",
    )
    core.sites[ModuleFaces.LEFT].attach_body(
        body=fl_leg.body,
        prefix="fl_leg",
    )
    
    fl_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fl_leg2.body,
        prefix="fl_flipper",
    )
    fl_leg2.sites[ModuleFaces.FRONT].attach_body(
        body=fl_flipper.body,
        prefix="fl_flipper2",
    )
    
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=fr_leg.body,
        prefix="fr_leg",
    )
    
    fr_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fr_leg2.body,
        prefix="fr_flipper",
    )
    fr_leg2.sites[ModuleFaces.FRONT].attach_body(
        body=fr_flipper.body,
        prefix="fr_flipper2",
    )
    
    butt.sites[ModuleFaces.LEFT].attach_body(
        body=bl_leg.body,
        prefix="bl_leg",
    )
    bl_leg.sites[ModuleFaces.FRONT].attach_body(
        body=bl_flipper.body,
        prefix="bl_flipper",
    )
    butt.sites[ModuleFaces.RIGHT].attach_body(
        body=br_leg.body,
        prefix="br_leg",
    )
    br_leg.sites[ModuleFaces.FRONT].attach_body(
        body=br_flipper.body,
        prefix="br_flipper",
    )
    return core
