#pragma once

#include <cstdint>
#include <string>

bool TimeSyncInit(std::string* out_error = nullptr);
uint64_t TimeSyncNowNs();
bool TimeSyncUsingPtp();
const std::string& TimeSyncClockSource();

void TimeSyncMarkCanSteerTxNs(uint64_t ts_ns);
void TimeSyncMarkCanBrakeTxNs(uint64_t ts_ns);
uint64_t TimeSyncGetCanSteerTxNs();
uint64_t TimeSyncGetCanBrakeTxNs();

