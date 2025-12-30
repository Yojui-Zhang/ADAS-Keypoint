#include "SortTracking.h"

#include "KeypointFilterSwitch.h"

#include <algorithm>
#include <cfloat>
#include <iterator>

using namespace std;

SORTTRACKING::SORTTRACKING() {}
SORTTRACKING::~SORTTRACKING() {}

// Computes IOU between two bounding boxes
double SORTTRACKING::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    const float inter = (bb_test & bb_gt).area();
    const float uni = bb_test.area() + bb_gt.area() - inter;
    if (uni < static_cast<float>(DBL_EPSILON)) return 0.0;
    return static_cast<double>(inter / uni);
}

std::vector<TrackingBox> SORTTRACKING::TrackingResult(const std::vector<Object>& bboxes)
{
    // Safety: detData is a member; ensure per-frame cleanliness.
    detData.clear();

    // 1) Convert detections -> TrackingBox (keep kpts)
    int det_id = 0;
    for (const auto& obj : bboxes)
    {
        TrackingBox temp;
        temp.id = det_id;
        temp.class_id = obj.class_id;
        temp.box = Rect_<float>(Point_<float>(obj.box.x, obj.box.y),
                                Point_<float>(obj.box.width + obj.box.x, obj.box.height + obj.box.y));
        temp.score = obj.score;
        temp.kpts = obj.kpts;
        detData.push_back(temp);
        ++det_id;
    }

    frame_count++;

    // Global keypoint smoother (EMA/KF switch)
    sort_kpt::KeypointFilterSwitch& kptFilter = sort_kpt::GlobalKeypointFilter();

    // 2) Bootstrap trackers on the first frame
    if (trackers.empty())
    {
        for (unsigned int i = 0; i < detData.size(); ++i)
        {
            KalmanTracker trk(detData[i].box, detData[i].score);
            trk.m_class = detData[i].class_id;

            const int tid = trk.m_id;
            trackers.push_back(trk);

            // Initialize kpt filter state immediately (so this frame has output)
            kptFilter.Update(tid, &detData[i].kpts);
        }

        // Output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end(); ++it)
        {
            TrackingBox res;
            res.box = it->get_state();
            res.id = it->m_id + 1; // keep existing +1 behavior
            res.class_id = it->m_class;
            res.frame = frame_count;
            res.score = it->score;

            std::vector<cv::Point3f> out_kpts;
            if (kptFilter.GetOutput(it->m_id, out_kpts)) res.kpts = out_kpts;
            else res.kpts.clear();

            frameTrackingResult.push_back(res);
        }

        return frameTrackingResult;
    }

    // 3) Predict step for all existing trackers
    predictedBoxes.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        Rect_<float> pBox = it->predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            ++it;
        }
        else
        {
            // Invalid track -> remove tracker and its kpt state
            kptFilter.Erase(it->m_id);
            it = trackers.erase(it);
        }
    }

    // 4) Data association: Hungarian on (1 - IOU)
    trkNum = predictedBoxes.size();
    detNum = detData.size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0.0));

    for (unsigned int i = 0; i < trkNum; ++i) {
        for (unsigned int j = 0; j < detNum; ++j) {
            iouMatrix[i][j] = 1.0 - GetIOU(predictedBoxes[i], detData[j].box);
        }
    }

    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();
    matchedPairs.clear();

    // 4.1) Find unmatched detections / trajectories
    if (detNum > trkNum)
    {
        for (unsigned int n = 0; n < detNum; ++n) allItems.insert(static_cast<int>(n));
        for (unsigned int i = 0; i < trkNum; ++i) if (assignment[i] != -1) matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int> >(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum)
    {
        for (unsigned int i = 0; i < trkNum; ++i) {
            if (assignment[i] == -1) unmatchedTrajectories.insert(static_cast<int>(i));
        }
    }

    // 4.2) Filter low-IOU matches
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) continue;

        const double iou = 1.0 - iouMatrix[i][assignment[i]];
        if (iou < iouThreshold)
        {
            unmatchedTrajectories.insert(static_cast<int>(i));
            unmatchedDetections.insert(assignment[i]);
        }
        else
        {
            matchedPairs.push_back(cv::Point(static_cast<int>(i), assignment[i]));
        }
    }

    // 5) Update matched trackers
    for (unsigned int i = 0; i < matchedPairs.size(); ++i)
    {
        const int trkIdx = matchedPairs[i].x;
        const int detIdx = matchedPairs[i].y;

        trackers[trkIdx].update(detData[detIdx].box, detData[detIdx].score);

        const int tid = trackers[trkIdx].m_id;
        kptFilter.Update(tid, &detData[detIdx].kpts);
    }

    // 6) Create new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        KalmanTracker tracker(detData[umd].box, detData[umd].score);
        tracker.m_class = detData[umd].class_id;

        const int tid = tracker.m_id;
        kptFilter.Update(tid, &detData[umd].kpts);

        trackers.push_back(tracker);
    }

    // 7) Unmatched trajectories: predict-only for keypoints (if already initialized)
    for (auto utrk : unmatchedTrajectories)
    {
        if (utrk >= 0 && utrk < static_cast<int>(trackers.size()))
        {
            const int tid = trackers[utrk].m_id;
            kptFilter.Update(tid, nullptr);
        }
    }

    // 8) Build output
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end(); ++it)
    {
        if ((it->m_time_since_update < 1) &&
            (it->m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = it->get_state();
            res.id = it->m_id + 1;
            res.class_id = it->m_class;
            res.frame = frame_count;
            res.score = it->score;

            std::vector<cv::Point3f> out_kpts;
            if (kptFilter.GetOutput(it->m_id, out_kpts)) res.kpts = out_kpts;
            else res.kpts.clear();

            frameTrackingResult.push_back(res);
        }
    }

    // 9) Remove dead tracks (and their keypoints)
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (it->m_time_since_update > max_age)
        {
            kptFilter.Erase(it->m_id);
            it = trackers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return frameTrackingResult;
}
