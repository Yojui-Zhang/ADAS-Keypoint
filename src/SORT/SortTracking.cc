#include "SortTracking.h"

#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace std;

// ==============================
// Keypoint filter switch (EMA / KF)  (本檔案私有)
// - 以 track_id 為鍵，維護每條軌跡的關鍵點平滑狀態
// - 允許用「開關」選擇 EMA 或 KF：
//   1) 編譯期預設：-DSORT_KPT_FILTER_DEFAULT_KF=1  (預設 0=EMA)
//   2) 執行期覆蓋：環境變數 SORT_KPT_FILTER = "kf" / "ema" / "1" / "0"
//   * 若同時設定，執行期（環境變數）優先。
// ==============================

#ifndef SORT_KPT_FILTER_DEFAULT_KF
#define SORT_KPT_FILTER_DEFAULT_KF 1
#endif

static inline bool _use_kpt_kf()
{
    static int cached = -1;
    if (cached != -1) return cached != 0;

#if SORT_KPT_FILTER_DEFAULT_KF
    cached = 1;
#else
    cached = 0;
#endif

    if (const char* env = std::getenv("SORT_KPT_FILTER")) {
        // 去前後空白（簡單處理）
        while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') ++env;
        if (std::strcmp(env, "kf") == 0 || std::strcmp(env, "KF") == 0 ||
            std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0) {
            cached = 1;
        } else if (std::strcmp(env, "ema") == 0 || std::strcmp(env, "EMA") == 0 ||
                   std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) {
            cached = 0;
        }
    }
    return cached != 0;
}

// 當前幀輸出的平滑後 kpts，供輸出時讀取
static std::unordered_map<int, std::vector<cv::Point3f>> __kpt_outputs;

// 夠用又安全的 clamp
template <typename T> static inline T _clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

// ------------------------------
// EMA implementation
// ------------------------------
struct _KptEMA {
    cv::Point2f pt{0.f, 0.f};
    bool initialized{false};
};

struct _TrackKptEMAState {
    vector<_KptEMA> kpts;   // 每個關鍵點的 EMA 狀態
    int num_kpts{0};
    int missed{0};
};

static std::unordered_map<int, _TrackKptEMAState> __kpt_ema_states;

// 以 conf 做自適應 EMA：conf 高 -> 反應快 (alpha 大)；conf 低 -> 更平滑 (alpha 小)
static inline float _conf_to_alpha(float conf, float conf_thr, float alpha_lo, float alpha_hi) {
    conf = _clamp(conf, 0.0f, 1.0f);
    float t = (conf <= conf_thr) ? 0.f : (conf - conf_thr) / (1.f - conf_thr);
    return alpha_lo + (alpha_hi - alpha_lo) * t;
}

static std::vector<cv::Point3f> _update_kpts_ema(int track_id, const std::vector<cv::Point3f>* meas,
                                                float conf_thr=0.2f, float alpha_lo=0.15f, float alpha_hi=0.7f)
{
    auto &st = __kpt_ema_states[track_id];

    // 初始化/調整長度
    if (meas && (int)meas->size() != st.num_kpts) {
        st.kpts.assign(meas->size(), _KptEMA{});
        st.num_kpts = (int)meas->size();
    } else if (!meas && st.num_kpts == 0) {
        // 沒有量測也沒有狀態，回空
        return {};
    }

    std::vector<cv::Point3f> out;
    out.resize(st.num_kpts, cv::Point3f(0.f, 0.f, 0.f));

    if (meas) {
        st.missed = 0;
        for (int i = 0; i < st.num_kpts; ++i) {
            const float mx = (*meas)[i].x;
            const float my = (*meas)[i].y;
            const float mc = (*meas)[i].z;  // keypoint conf (0~1 或 0~100)

            float conf = mc;
            if (conf > 1.5f) conf = conf / 100.f; // 兼容 0~100
            conf = _clamp(conf, 0.f, 1.f);

            // 僅對有效量測做更新
            if (conf > 0.f && mx > 0.f && my > 0.f) {
                float alpha = _conf_to_alpha(conf, conf_thr, alpha_lo, alpha_hi);
                if (!st.kpts[i].initialized) {
                    st.kpts[i].pt = cv::Point2f(mx, my);
                    st.kpts[i].initialized = true;
                } else {
                    st.kpts[i].pt.x = alpha * mx + (1.f - alpha) * st.kpts[i].pt.x;
                    st.kpts[i].pt.y = alpha * my + (1.f - alpha) * st.kpts[i].pt.y;
                }
                out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, mc);
            } else {
                // 量測無效，沿用上一幀（若尚未初始化則保持 (0,0,0)）
                if (st.kpts[i].initialized) {
                    out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, 0.f);
                }
            }
        }
    } else {
        // 沒有量測：保持 last（z=0），missed++
        st.missed += 1;
        for (int i = 0; i < st.num_kpts; ++i) {
            if (st.kpts[i].initialized) {
                out[i] = cv::Point3f(st.kpts[i].pt.x, st.kpts[i].pt.y, 0.f);
            }
        }
    }
    return out;
}

// ------------------------------
// Keypoint KF implementation (2D constant-velocity)
// - state: [x, y, vx, vy]
// - measurement: [x, y]
// - v(conf) 用於：
//   1) v < conf_thr -> predict only (缺測/低可信)
//   2) v -> 動態調整 measurement noise R (v 高 -> R 小)
// - 內建 gating：量測距離預測太遠 -> 拒收
// ------------------------------
struct _KptKF {
    float x[4]{0.f, 0.f, 0.f, 0.f};     // [x,y,vx,vy]
    float P[4][4]{};                    // covariance
    bool initialized{false};

    void init(float mx, float my, float init_pos_var, float init_vel_var) {
        x[0] = mx; x[1] = my; x[2] = 0.f; x[3] = 0.f;
        for (int i=0;i<4;++i) for (int j=0;j<4;++j) P[i][j]=0.f;
        P[0][0] = init_pos_var;
        P[1][1] = init_pos_var;
        P[2][2] = init_vel_var;
        P[3][3] = init_vel_var;
        initialized = true;
    }

    void predict(float q) {
        // F * x
        x[0] = x[0] + x[2];
        x[1] = x[1] + x[3];

        // P = F P F^T + Q, where F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
        static const float F[4][4] = {
            {1,0,1,0},
            {0,1,0,1},
            {0,0,1,0},
            {0,0,0,1}
        };

        float FP[4][4]{};
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                float s=0.f;
                for (int k=0;k<4;++k) s += F[i][k]*P[k][j];
                FP[i][j]=s;
            }
        }
        float Pp[4][4]{};
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                float s=0.f;
                for (int k=0;k<4;++k) s += FP[i][k]*F[j][k]; // F^T
                Pp[i][j]=s;
            }
        }
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) P[i][j]=Pp[i][j];
            P[i][i] += q;
        }
    }

    bool correct(float mx, float my, float r, float gate_dist_px) {
        // gating on predicted position
        const float dx = mx - x[0];
        const float dy = my - x[1];
        const float dist2 = dx*dx + dy*dy;
        if (gate_dist_px > 0.f && dist2 > gate_dist_px*gate_dist_px) {
            return false; // reject outlier
        }

        // S = HPH^T + R; H selects x,y -> S = [[P00,P01],[P10,P11]] + rI
        float S00 = P[0][0] + r;
        float S01 = P[0][1];
        float S10 = P[1][0];
        float S11 = P[1][1] + r;

        // inv(S) for 2x2
        float det = S00*S11 - S01*S10;
        if (std::fabs(det) < 1e-9f) return false;
        float invS00 =  S11 / det;
        float invS01 = -S01 / det;
        float invS10 = -S10 / det;
        float invS11 =  S00 / det;

        // PH^T : 4x2 matrix = [P[:,0], P[:,1]]
        float K[4][2]{};
        for (int i=0;i<4;++i) {
            float PHt0 = P[i][0];
            float PHt1 = P[i][1];
            K[i][0] = PHt0*invS00 + PHt1*invS10;
            K[i][1] = PHt0*invS01 + PHt1*invS11;
        }

        // innovation y = z - Hx
        float y0 = mx - x[0];
        float y1 = my - x[1];

        // x = x + K y
        for (int i=0;i<4;++i) {
            x[i] = x[i] + K[i][0]*y0 + K[i][1]*y1;
        }

        // P = (I - K H) P, where KH affects only first two columns
        float I_KH[4][4]{};
        for (int i=0;i<4;++i) for (int j=0;j<4;++j) I_KH[i][j] = (i==j) ? 1.f : 0.f;
        for (int i=0;i<4;++i) {
            I_KH[i][0] -= K[i][0];
            I_KH[i][1] -= K[i][1];
        }
        float Pnew[4][4]{};
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                float s=0.f;
                for (int k=0;k<4;++k) s += I_KH[i][k]*P[k][j];
                Pnew[i][j]=s;
            }
        }
        for (int i=0;i<4;++i) for (int j=0;j<4;++j) P[i][j]=Pnew[i][j];
        return true;
    }
};

struct _TrackKptKFState {
    std::vector<_KptKF> kfs;
    int num_kpts{0};
};

static std::unordered_map<int, _TrackKptKFState> __kpt_kf_states;

static std::vector<cv::Point3f> _update_kpts_kf(int track_id, const std::vector<cv::Point3f>* meas,
                                               float conf_thr=0.2f,
                                               float process_var=1.0f,
                                               float meas_var_base=25.0f,
                                               float gate_dist_px=80.0f,
                                               float init_pos_var=100.0f,
                                               float init_vel_var=1000.0f)
{
    auto &st = __kpt_kf_states[track_id];

    // 初始化/調整長度
    if (meas && (int)meas->size() != st.num_kpts) {
        st.kfs.assign(meas->size(), _KptKF{});
        st.num_kpts = (int)meas->size();
    } else if (!meas && st.num_kpts == 0) {
        return {};
    }

    std::vector<cv::Point3f> out(st.num_kpts, cv::Point3f(0.f, 0.f, 0.f));

    // 先 predict（有無量測都要 predict）
    for (int i=0;i<st.num_kpts;++i) {
        if (st.kfs[i].initialized) st.kfs[i].predict(process_var);
    }

    if (!meas) {
        // 缺測：輸出 predict 結果（z=0）
        for (int i=0;i<st.num_kpts;++i) {
            if (st.kfs[i].initialized) {
                out[i] = cv::Point3f(st.kfs[i].x[0], st.kfs[i].x[1], 0.f);
            }
        }
        return out;
    }

    // 有量測：逐點更新（或拒收）
    for (int i=0;i<st.num_kpts;++i) {
        const float mx = (*meas)[i].x;
        const float my = (*meas)[i].y;
        const float mc = (*meas)[i].z;

        float conf = mc;
        if (conf > 1.5f) conf = conf / 100.f; // 兼容 0~100
        conf = _clamp(conf, 0.f, 1.f);

        const bool meas_valid = (conf >= conf_thr) && (mx > 0.f) && (my > 0.f);

        if (!st.kfs[i].initialized) {
            if (meas_valid) {
                st.kfs[i].init(mx, my, init_pos_var, init_vel_var);
                out[i] = cv::Point3f(mx, my, mc);
            } else {
                // 未初始化且缺測：保持 0
                out[i] = cv::Point3f(0.f, 0.f, 0.f);
            }
            continue;
        }

        if (!meas_valid) {
            // 低可信：predict-only
            out[i] = cv::Point3f(st.kfs[i].x[0], st.kfs[i].x[1], 0.f);
            continue;
        }

        // conf 高 -> R 小；conf 低(但>=thr) -> R 大
        float r = meas_var_base / std::max(conf, 0.05f);
        r = _clamp(r, 1.0f, 1e4f);

        bool accepted = st.kfs[i].correct(mx, my, r, gate_dist_px);
        if (accepted) out[i] = cv::Point3f(st.kfs[i].x[0], st.kfs[i].x[1], mc);
        else          out[i] = cv::Point3f(st.kfs[i].x[0], st.kfs[i].x[1], 0.f);
    }
    return out;
}

// 統一入口：依開關選 EMA / KF
static inline std::vector<cv::Point3f> _update_kpts(int track_id, const std::vector<cv::Point3f>* meas)
{
    if (_use_kpt_kf()) return _update_kpts_kf(track_id, meas);
    return _update_kpts_ema(track_id, meas);
}

// 當追蹤器被刪除時，清掉對應的 kpt 狀態
static inline void _erase_kpt_state(int track_id) {
    __kpt_ema_states.erase(track_id);
    __kpt_kf_states.erase(track_id);
    __kpt_outputs.erase(track_id);
}

// ==============================
// 原有 SORTTRACKING 實作，僅插入 kpts 的處理
// ==============================

SORTTRACKING::SORTTRACKING() {}
SORTTRACKING::~SORTTRACKING() {}

// Computes IOU between two bounding boxes
double SORTTRACKING::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) return 0;
    return (double)(in / un);
}

std::vector<TrackingBox> SORTTRACKING::TrackingResult(const std::vector<Object>& bboxes)
{
    int input_i = 0;

    // 將檢測結果轉為 TrackingBox，並把 kpts 一併放入 detData（原本沒放）
    for (const auto& obj : bboxes)
    {
        TrackingBox temp;
        temp.id = input_i;
        temp.class_id = obj.class_id;
        temp.box = Rect_<float>(Point_<float>(obj.box.x, obj.box.y),
                                Point_<float>(obj.box.width + obj.box.x, obj.box.height + obj.box.y));
        temp.score = obj.score;
        temp.kpts  = obj.kpts;  // <<< 新增：帶入該檢測的 keypoints
        detData.push_back(temp);
        input_i++;
    }

    input_i = 0;
    frame_count++;

    if (trackers.size() == 0) // 第一幀
    {
        // 用第一次檢測初始化 Kalman 追蹤器 + 初始化 kpt EMA 狀態
        for (unsigned int i = 0; i < detData.size(); i++)
        {
            KalmanTracker trk = KalmanTracker(detData[i].box, detData[i].score);
            trk.m_class = detData[i].class_id;
            // 先建立 tracker
            trackers.push_back(trk);

            // 初始化對應的 kpts 狀態（使用 KalmanTracker::m_id 作為 key）
            int tid = trk.m_id;
            auto out_k = _update_kpts(tid, &detData[i].kpts);
            __kpt_outputs[tid] = std::move(out_k);
        }
    }
    else
    {
        // === 原有 predict ===
        predictedBoxes.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                // 追蹤器無效 -> 一併清掉 kpt 狀態
                _erase_kpt_state((*it).m_id);
                it = trackers.erase(it);
            }
        }

        // === 原有 Hungarian 匹配（以 1 - IOU 作成本）===
        trkNum = predictedBoxes.size();
        detNum = detData.size();
        iouMatrix.clear();
        iouMatrix.resize(trkNum, vector<double>(detNum, 0));
        for (unsigned int i = 0; i < trkNum; i++)
            for (unsigned int j = 0; j < detNum; j++)
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);

        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum)
        {
            for (unsigned int n = 0; n < detNum; n++) allItems.insert(n);
            for (unsigned int i = 0; i < trkNum; ++i) matchedItems.insert(assignment[i]);
            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum)
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1)
                    unmatchedTrajectories.insert(i);
        }

        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        // === 原有：更新匹配到的追蹤器 ===
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            int trkIdx = matchedPairs[i].x;
            int detIdx = matchedPairs[i].y;

            trackers[trkIdx].update(detData[detIdx].box, detData[detIdx].score);

            // 新增：用此 detection 的 kpts 更新 EMA，並保存當前幀輸出
            int tid = trackers[trkIdx].m_id;
            auto out_k = _update_kpts(tid, &detData[detIdx].kpts);
            __kpt_outputs[tid] = std::move(out_k);
        }

        // === 原有：未匹配的檢測 -> 新增追蹤器 ===
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detData[umd].box, detData[umd].score);
            tracker.m_class = detData[umd].class_id;
            int tid = tracker.m_id;  // 在 push_back 前即可取到 id（見 kalman.h 建構子）

            // 先初始化 kpts EMA，確保本幀就有輸出
            auto out_k = _update_kpts(tid, &detData[umd].kpts);
            __kpt_outputs[tid] = std::move(out_k);

            trackers.push_back(tracker);
        }

        // === 新增：未匹配的軌跡（本幀沒有量測）也要生成 kpts 輸出（沿用 last, z=0）===
        for (auto utrk : unmatchedTrajectories)
        {
            if (utrk >= 0 && utrk < (int)trackers.size()) {
                int tid = trackers[utrk].m_id;
                auto out_k = _update_kpts(tid, nullptr);
                __kpt_outputs[tid] = std::move(out_k);
            }
        }
    }

    // === 原有：彙整輸出（這裡把平滑後的 kpts 帶進去）===
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;     // 保持原有 +1 輸出
            res.class_id = (*it).m_class;
            res.frame = frame_count;
            res.score = (*it).score;

            // 取出上一段流程存好的平滑後 kpts
            auto found = __kpt_outputs.find((*it).m_id);
            if (found != __kpt_outputs.end())
                res.kpts = found->second;   // 平滑後
            else
                res.kpts.clear();

            frameTrackingResult.push_back(res);
            it++;
        }
        else
        {
            it++;
        }

        // 原有：刪除過時追蹤器；新增：同步刪除 kpt 狀態
        if (it != trackers.end() && (*it).m_time_since_update > max_age) {
            _erase_kpt_state((*it).m_id);
            it = trackers.erase(it);
        }
    }

    detData.clear();
    return frameTrackingResult;
}