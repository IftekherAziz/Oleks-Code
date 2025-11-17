import numpy as np

from core.geometry import radial_vector


experiments = [
        # A.1: same tangential direction
        {
            "name": "A.1",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-1, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([0, 1, 0]),
            "compute_pB": lambda angle_deg: np.array([
                np.cos(np.radians(angle_deg)),
                np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vB": lambda angle_deg: np.array([0, 1, 0]),
            "title": "A.1 - same tangential direction",
            "xlabel": "Angle of the observer (deg)",
            "ylabel": "Distance"
        },
        # A.2: same radial direction
        {
            "name": "A.2",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-1, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([-1, 0, 0]),
            "compute_pB": lambda angle_deg: np.array([
                np.cos(np.radians(angle_deg)),
                np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vB": lambda angle_deg: np.array([-1, 0, 0]),
            "title": "A.2 - same radial direction",
            "xlabel": "Angle of the observer (deg)",
            "ylabel": "Distance"
        },
        # A.3: both stars move, same direction
        {
            "name": "A.3",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([
                2 * np.cos(np.radians(angle_deg)),
                2 * np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vA": lambda angle_deg: np.array([0, 1, 0]),
            "compute_pB": lambda angle_deg: np.array([
                (2 + 0.3) * np.cos(np.radians(angle_deg)),
                (2 + 0.3) * np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vB": lambda angle_deg: np.array([0, 1, 0]),
            "title": "A.3 - both stars move, same direction",
            "xlabel": "Angle of the observer (deg)",
            "ylabel": "Distance"
        },
        # A.4: both stars move, opposite direction
        {
            "name": "A.4",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([
                2 * np.cos(np.radians(angle_deg)),
                2 * np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vA": lambda angle_deg: np.array([0, 1, 0]),
            "compute_pB": lambda angle_deg: np.array([
                (2 + 0.3) * np.cos(np.radians(angle_deg)),
                (2 + 0.3) * np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vB": lambda angle_deg: np.array([0, -1, 0]),
            "title": "A.4 - both stars move, opposite direction",
            "xlabel": "Angle of the observer (deg)",
            "ylabel": "Distance"
        },
        # B.1: position A fixed, vA radial, vB rotating
        {
            "name": "B.1",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-2, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([-1, 0, 0]),
            "compute_pB": lambda angle_deg: np.array([-2, 0.3, 0]),
            "compute_vB": lambda angle_deg: np.array([
                np.cos(np.radians(angle_deg)),
                np.sin(np.radians(angle_deg)),
                0
            ]),
            "title": "B.1 - position stays the same, radial direction",
            "xlabel": "Angle of the velocity vector (deg)",
            "ylabel": "Distance"
        },
        # B.2: position A fixed, vA tangential, vB rotating
        {
            "name": "B.2",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-2, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([0, 1, 0]),
            "compute_pB": lambda angle_deg: np.array([-2, 0.3, 0]),
            "compute_vB": lambda angle_deg: np.array([
                np.cos(np.radians(angle_deg)),
                np.sin(np.radians(angle_deg)),
                0
            ]),
            "title": "B.2 - position stays the same, tangential direction",
            "xlabel": "Angle of the velocity vector (deg)",
            "ylabel": "Distance"
        },
        # B.3: positions differ, vA tangential at A, vB rotating at B
        {
            "name": "B.3",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-2, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([0, 1, 0]),
            "compute_pB": lambda angle_deg: np.array([0, 2, 0]),
            "compute_vB": lambda angle_deg: np.array([
                np.cos(np.radians(angle_deg)),
                np.sin(np.radians(angle_deg)),
                0
            ]),
            "title": "B.3 - different positions, velocity changes, tangential direction",
            "xlabel": "Angle of the velocity vector (deg)",
            "ylabel": "Distance"
        },
        # B.4: positions differ, vA radial at A, vB rotating at B
        {
            "name": "B.4",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-2, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([-1, 0, 0]),
            "compute_pB": lambda angle_deg: np.array([2, 0, 0]),
            "compute_vB": lambda angle_deg: np.array([
                np.cos(np.radians(angle_deg)),
                np.sin(np.radians(angle_deg)),
                0
            ]),
            "title": "B.4 - different positions, velocity changes, radial direction",
            "xlabel": "Angle of the velocity vector (deg)",
            "ylabel": "Distance"
        },
        # C.1: A fixed, vA radial; B on a circle, vB = radial at B
        {
            "name": "C.1",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-2, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([-1, 0, 0]),
            "compute_pB": lambda angle_deg: np.array([
                2.3 * np.cos(np.radians(angle_deg)),
                2.3 * np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vB": lambda angle_deg: radial_vector(
                np.array([
                    2.3 * np.cos(np.radians(angle_deg)),
                    2.3 * np.sin(np.radians(angle_deg)),
                    0
                ])
            ),
            "title": "C.1 - position and velocity change, radial",
            "xlabel": "Angle of change (deg)",
            "ylabel": "Distance"
        },
        # C.2: A fixed, vA tangential at A; B on a circle, vB tangential at B
        {
            "name": "C.2",
            "x_values": np.degrees(np.linspace(0, 2 * np.pi, 100)),
            "compute_pA": lambda angle_deg: np.array([-2, 0, 0]),
            "compute_vA": lambda angle_deg: np.array([0, 1, 0]),
            "compute_pB": lambda angle_deg: np.array([
                2.3 * np.cos(np.radians(angle_deg)),
                2.3 * np.sin(np.radians(angle_deg)),
                0
            ]),
            "compute_vB": lambda angle_deg: np.array([
                - (2.3 * np.sin(np.radians(angle_deg))),
                (2.3 * np.cos(np.radians(angle_deg))),
                0
            ]),
            "title": "C.2 - position and velocity change, tangential",
            "xlabel": "Angle of change (deg)",
            "ylabel": "Distance"
        },
        # D.1: A & B fixed positions; vA fixed dir, vB direction fixed, speed varies
        {
            "name": "D.1",
            "x_values": np.linspace(0.1, 4, 100),
            "compute_pA": lambda speed: np.array([-2, 0, 0]),
            "compute_vA": lambda speed: np.array([-1, 0, 0]),
            "compute_pB": lambda speed: np.array([-2, 0.3, 0]),
            "compute_vB": lambda speed: np.array([-1, 0, 0]) * speed,
            "title": "D.1 - Fixed direction, B's speed varies",
            "xlabel": "Speed of Star B (km/s)",
            "ylabel": "Distance"
        },
        # D.2: A & B fixed; vA = radial at A, vB = radial at B, speed varies
        {
            "name": "D.2",
            "x_values": np.linspace(0.1, 4, 100),
            "compute_pA": lambda speed: np.array([-2, 0, 0]),
            "compute_vA": lambda speed: radial_vector(np.array([-2, 0, 0])),
            "compute_pB": lambda speed: np.array([-2, 0.3, 0]),
            "compute_vB": lambda speed: radial_vector(np.array([-2, 0.3, 0])) * speed,
            "title": "D.2 - fixed radial direction, B's speed varies",
            "xlabel": "Speed of B (km/s)",
            "ylabel": "Distance"
        },
        # D.3: A & B fixed; vA = tangential at A, vB = tangential at B, speed varies
        {
            "name": "D.3",
            "x_values": np.linspace(0.1, 2, 100),
            "compute_pA": lambda speed: np.array([-2, 0, 0]),
            "compute_vA": lambda speed: np.array([0, 1, 0]),
            "compute_pB": lambda speed: np.array([-2, 0.3, 0]),
            "compute_vB": lambda speed: np.array([
                -radial_vector(np.array([-2, 0.3, 0]))[1],
                radial_vector(np.array([-2, 0.3, 0]))[0],
                0
            ]) * speed,
            "title": "D.3 - fixed tangential direction, B's speed varies",
            "xlabel": "Speed of B (km/s)",
            "ylabel": "Distance"
        },
        # D.4: A & B fixed; vA arbitrary, vB arbitrary, speed varies
        {
            "name": "D.4",
            "x_values": np.linspace(0.1, 2, 100),
            "compute_pA": lambda speed: np.array([-2, 0, 0]),
            "compute_vA": lambda speed: np.array([-1, 0.5, 0]),
            "compute_pB": lambda speed: np.array([-2, 0.3, 0]),
            "compute_vB": lambda speed: np.array([-0.5, 0.5, 0]) * speed,
            "title": "D.4 - mostly radial, B's speed varies",
            "xlabel": "Speed of B (km/s)",
            "ylabel": "Distance"
        },
        # D.5: A & B fixed; vA arbitrary, vB arbitrary, speed varies
        {
            "name": "D.5",
            "x_values": np.linspace(0.1, 2, 100),
            "compute_pA": lambda speed: np.array([-2, 0, 0]),
            "compute_vA": lambda speed: np.array([0.5, 0.5, 0]),
            "compute_pB": lambda speed: np.array([-2, 0.3, 0]),
            "compute_vB": lambda speed: np.array([0.5, 0.7, 0]) * speed,
            "title": "D.5 - mostly tangential, B's speed varies",
            "xlabel": "Speed of B (km/s)",
            "ylabel": "Distance"
        },
    ]