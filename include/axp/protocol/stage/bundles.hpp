#pragma once

#include "../../bundles/token_bundles.hpp"

namespace axp::protocol::stage {

struct SlotHandle {
    static constexpr auto id = iro::util::fnv1a_64_cstr("axp.stage.slot_handle");
};

template<class SlotSubj, class Lifetime, long long Bytes>
using ProducerIssued = axp::bundle::SmemFillingTx<SlotSubj, Lifetime, Bytes>;

template<class SlotSubj, class Lifetime, long long Bytes>
using ProducerCommittedTx = axp::bundle::SmemCommittedTx<SlotSubj, Lifetime, Bytes>;

template<class SlotSubj, class ExecGroup, class Lifetime, long long Bytes>
using ProducerCommitted = axp::bundle::SmemReadyTx<SlotSubj, ExecGroup, Lifetime, Bytes>;

template<class SlotSubj, class Lifetime>
using ConsumerConsumed = axp::bundle::SmemConsumed<SlotSubj, Lifetime>;

template<class SlotSubj, class Lifetime>
using SlotReleased = axp::bundle::SmemReleased<SlotSubj, Lifetime>;
}
