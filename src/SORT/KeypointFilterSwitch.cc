#include "KeypointFilterSwitch.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#ifndef SORT_KPT_FILTER_DEFAULT_KF
#define SORT_KPT_FILTER_DEFAULT_KF 1
/*
Alse can use 
    export SORT_KPT_FILTER=ema
To change this state!
*/
#endif

namespace sort_kpt {

static inline const char* skip_ws_(const char* s)
{
    if (!s) return s;
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') ++s;
    return s;
}

KeypointFilterSwitch::KeypointFilterSwitch()
    : active_(ResolveActive_()), ema_(), kf_()
{
}

KeypointFilterSwitch::FilterType KeypointFilterSwitch::ResolveActive_()
{

    // Compile-time default
#if SORT_KPT_FILTER_DEFAULT_KF
    FilterType chosen = KF;
#else
    FilterType chosen = EMA;
#endif

    // Runtime override via env var
    const char* env = std::getenv("SORT_KPT_FILTER");
    env = skip_ws_(env);
    if (!env || *env == '\0') {

        std::cerr << ">>>> [Tips] SORT kpt choose: " 
              << (chosen == KF ? "KF" : "EMA") << std::endl;

        return chosen;
    }

    if (std::strcmp(env, "kf") == 0 || std::strcmp(env, "KF") == 0 ||
        std::strcmp(env, "1") == 0  || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0) {
        chosen = KF;
    } else if (std::strcmp(env, "ema") == 0 || std::strcmp(env, "EMA") == 0 ||
               std::strcmp(env, "0") == 0   || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) {
        chosen = EMA;
    }

    std::cerr << ">>>> [Tips] SORT kpt choose: " 
              << (chosen == KF ? "KF" : "EMA") << std::endl;

    return chosen;
}

void KeypointFilterSwitch::Update(int track_id, const std::vector<cv::Point3f>* meas)
{
    if (active_ == KF) kf_.Update(track_id, meas);
    else              ema_.Update(track_id, meas);
}

bool KeypointFilterSwitch::GetOutput(int track_id, std::vector<cv::Point3f>& out) const
{
    if (active_ == KF) return kf_.GetOutput(track_id, out);
    return ema_.GetOutput(track_id, out);
}

void KeypointFilterSwitch::Erase(int track_id)
{
    // Always erase both to be safe (e.g., if user switches filter at runtime between runs).
    ema_.Erase(track_id);
    kf_.Erase(track_id);
}

void KeypointFilterSwitch::Clear()
{
    ema_.Clear();
    kf_.Clear();
}

KeypointFilterSwitch& GlobalKeypointFilter()
{
    static KeypointFilterSwitch inst;
    return inst;
}

} // namespace sort_kpt
