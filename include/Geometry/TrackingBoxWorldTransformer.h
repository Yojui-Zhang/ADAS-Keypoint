#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "config.h"
#include "WorldProjector.h"

class TrackingBoxWorldTransformer
{
public:
    explicit TrackingBoxWorldTransformer(WorldProjector projector)
    : projector_(std::move(projector)) {}

    // Requirement:
    //   input: TrackingBox
    //   output: vector<cv::Point3f>
    //
    // class_id == 0: transform kpts -> world points
    // class_id  > 0: transform bottom-left & bottom-right of box -> world points (size = 2)
    std::vector<cv::Point3f> toWorldPoints(TrackingBox& tb) const;

private:
    WorldProjector projector_;
};
