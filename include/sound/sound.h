#pragma once

namespace sound {

// Requests the lane-departure warning sound without blocking the frame loop.
// Returns false when a previous warning sound is still in its cooldown window.
bool RequestLaneDepartureWarningSound();

}  // namespace sound
