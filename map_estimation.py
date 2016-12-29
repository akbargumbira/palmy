# coding=utf-8
import math
from palm import Palm, PalmModel
from extract_image import extract_image
from utilities import rotate_points, rotate_point


def iterative_map(image):
    palm_contour, palm_center, palm_radius, fingertips, rotation = extract_image(image)
    observed_palm = Palm(fingertips, palm_center, palm_radius, rotation)
    # Rotate first the model for faster projection
    r_fingertips = rotate_points(
        PalmModel.fingertips_location, PalmModel.palm_center, rotation)
    final_hypothesis = Palm(r_fingertips, PalmModel.palm_center,
                            PalmModel.palm_radius, rotation)
    post = posterior(observed_palm, final_hypothesis)
    min_posterior = 80
    while post > min_posterior:
        adm_hypothesis = admissible_instantiations(final_hypothesis)
        for hypothesis in adm_hypothesis:
            this_posterior = posterior(observed_palm, hypothesis)
            if this_posterior < post:
                post = this_posterior
                final_hypothesis = hypothesis
                print post
    return final_hypothesis


def posterior(observed_palm, hypothesis):
    """Return the posteriori probability."""
    penalty = 0
    for i, fingertip in enumerate(hypothesis.fingertips_location):
        penalty += math.hypot(
            hypothesis.fingertips_location[i][0] - observed_palm.fingertips_location[i][0],
            hypothesis.fingertips_location[i][1] - observed_palm.fingertips_location[i][1])
    penalty += math.hypot(
        hypothesis.palm_center[0] - observed_palm.palm_center[0],
        hypothesis.palm_center[1] - observed_palm.palm_center[1])

    # P = exp(-penalty)
    return penalty


def admissible_instantiations(hypothesis):
    """Generate admissible instantiations from given hypothesis."""
    instantiations = []

    thumb_shifts = admissible_shifts(hypothesis.fingertips_location[0])
    index_shifts = admissible_shifts(hypothesis.fingertips_location[1])
    middle_shifts = admissible_shifts(hypothesis.fingertips_location[2])
    ring_shifts = admissible_shifts(hypothesis.fingertips_location[3])
    baby_shifts = admissible_shifts(hypothesis.fingertips_location[4])
    center_shifts = admissible_shifts(hypothesis.palm_center)
    for thumb_shift in thumb_shifts:
        for index_shift in index_shifts:
            for middle_shift in middle_shifts:
                for ring_shift in ring_shifts:
                    for baby_shift in baby_shifts:
                        for center_shift in center_shifts:
                            fingertips = [thumb_shift, index_shift,
                                          middle_shift, ring_shift, baby_shift]
                            hypothesis = Palm(fingertips, center_shift)
                            instantiations.append(hypothesis)

    return instantiations


def admissible_shifts(point):
    frame_size = 3
    step = 5
    points = []
    for x in range((-frame_size + 1) / 2, (frame_size + 1) / 2):
        for y in range((-frame_size + 1) / 2, (frame_size + 1) / 2):
            dx = step * x
            dy = step * y
            points.append((point[0]+dx, point[1]+dy))
    return points





