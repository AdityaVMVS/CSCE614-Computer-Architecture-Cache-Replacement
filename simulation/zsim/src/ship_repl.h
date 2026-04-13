#ifndef SHIP_REPL_H_
#define SHIP_REPL_H_

#include "repl_policies.h"

// SHiP-PC replacement policy built on top of SRRIP
// -------------------------------------------------
// Per line:
//   - rrpv[id]         : SRRIP re-reference prediction value
//   - lineSig[id]      : signature for this line
//   - lineOutcome[id]  : 0 if never hit, 1 after first hit
//
// Global:
//   - shct[ ]          : Signature History Counter Table
//                        small saturating counters per signature
//
// Policy:
//   On hit:
//      rrpv[id] = 0 (hit promotion)
//      if first hit (outcome == 0), SHCT[ sig ]++ and outcome = 1
//
//   On eviction (victim selection in rank()):
//      if outcome == 0, SHCT[ sig ]--
//      if outcome == 1, do nothing
//
//   On insertion (in replaced()):
//      sig = signature of miss (taken from lastSig)
//      if SHCT[ sig ] == 0  → insert rrpv = rpvMax (3)
//      else                 → insert rrpv = rpvMax - 1 (2)
//      outcome = 0
// -------------------------------------------------

class SHiPReplPolicy : public ReplPolicy {
protected:
    uint32_t numLines; // total number of cache lines in this cache.
    uint32_t rpvMax; // maximum RRIP value (for 2 bit RRIP it is 3)
    bool     hitPromote; // controls hit behavior (true means on hit we set RRPV to 0)

    uint32_t* rrpv; // an array, one entry per line, storing the RRIP value
    uint64_t* lineSig; // an array of 64 bit signatures for each line
    uint8_t*  lineOutcome; // an array of 1 byte values, 0 or 1, telling if the line ever got a hit

    static const uint32_t SHCT_SIZE = 16384; //size of the Signature History Counter Table - 16K entries
    static const uint8_t  SHCT_MAX  = 7;    // maximum value of a counter in SHCT, 7 (b'111) means 3 bits
    uint8_t* shct; // array of counters indexed by signature

    uint64_t lastSig;          // signature for incoming miss
    uint32_t lastReplaced;     // last replaced line (must skip update on this)

    // -------------------------------------------------------------------
    // hashSig():
    //   Input  : a 64-bit signature value
    //   Output : a 32-bit scrambled (hashed) version of that signature
    //
    // Why do we need this?
    //   - SHCT is small (16K entries).
    //   - Many signatures may look similar.
    //   - Hashing mixes the bits so different signatures spread out
    //     across the table instead of clustering together.
    // -------------------------------------------------------------------
    inline uint32_t hashSig(uint64_t x) const {
        x ^= x >> 33;                          // mix high and low bits
        x *= 0xff51afd7ed558ccdULL;            // multiply by a large constant
        x ^= x >> 33;                          // mix again
        return (uint32_t)x;                    // return lower 32 bits
    }


    // -------------------------------------------------------------------
    // shctIndex():
    //   Input  : a 64-bit signature value
    //   Output : a valid index inside the SHCT table (0 .. SHCT_SIZE-1)
    //
    //   Uses hashSig() then masks with SHCT_SIZE - 1. Because SHCT_SIZE
    //   is a power of two, this maps signatures uniformly into the table.
    // -------------------------------------------------------------------
    inline uint32_t shctIndex(uint64_t sig) const {
        return hashSig(sig) & (SHCT_SIZE - 1);
    }

    // -------------------------------------------------------------------
    // getSignature():
    //   Purpose:
    //     Build a signature that represents the type of memory access
    //     causing the miss. SHiP uses this to group similar accesses and
    //     learn whether they tend to be useful or useless.
    //
    //   Design for this zsim setup:
    //     - Does not use program counter.
    //     - Uses a memory region based signature (SHiP-Mem style).
    //     - Learns only from demand data GETS / GETX requests.
    //     - Ignores instruction fetches and other access types.
    //
    //   Fields used:
    //     - lineAddr : cache line address (64 B granularity).
    //     - srcId    : core or requester id.
    //
    //   How the signature is formed:
    //     - Group lines into 16 KB regions (256 lines per region).
    //     - Use the region id as the base signature.
    //     - Mix in a few bits of srcId to separate cores if needed.
    //
    //   Result:
    //     - Many distinct signatures across the address space.
    //     - No dependency on prefetch flags.
    // -------------------------------------------------------------------
    inline uint64_t getSignature(const MemReq* req) const {
        if (!req) return 0;

        // Learn only from demand data accesses (loads and stores that fetch data)
        if (!(req->type == GETS || req->type == GETX)) {
            return 0;   // treat other traffic as "no signature"
        }

        // Ignore instruction fetches for SHiP training
        if (req->is(MemReq::IFETCH)) {
            return 0;
        }

        // lineAddr is already a line address (64 byte granularity)
        uint64_t line = req->lineAddr;

        // Group lines into 16 KB regions: 16 KB / 64 B = 256 lines, shift by 8
        const int REGION_SHIFT = 8;
        uint64_t region = line >> REGION_SHIFT;

        uint64_t sig = 0;

        // Lower bits: region id
        sig |= region;

        // Add a few bits of core id in higher bits (future multicores)
        sig |= (uint64_t(req->srcId) & 0xF) << 48;

        return sig;
    }

    // -------------------------------------------------------------------
    // trainEviction(): called when we evict a valid line with id
    // -------------------------------------------------------------------
    inline void trainEviction(uint32_t id) {
        // first checks if the line is valid
        if (!cc || !cc->isValid(id)) return;

        // If this line never had a hit (lineOutcome[id] == 0) 
        // then this signature is "bad", so we decrement the SHCT entry, 
        // but not below zero.

        if (lineSig[id] == 0) return;

        if (lineOutcome[id] == 0) {
            uint32_t idx = shctIndex(lineSig[id]);
            if (shct[idx] > 0) shct[idx]--;
        }
    }

public:

    SHiPReplPolicy(uint32_t _numLines, uint32_t _rpvMax, bool _hitPromote = true)
        : numLines(_numLines),
          rpvMax(_rpvMax),
          hitPromote(_hitPromote),
          lastSig(0),
          lastReplaced(UINT32_MAX)
    {
        rrpv        = gm_calloc<uint32_t>(numLines);
        lineSig     = gm_calloc<uint64_t>(numLines);
        lineOutcome = gm_calloc<uint8_t>(numLines);

        uint32_t initVal = (rpvMax > 0) ? (rpvMax - 1) : 0;

        for (uint32_t i = 0; i < numLines; i++) {
            rrpv[i] = initVal; // If valid, then all lines start with rpvMax - 1
            lineSig[i] = 0;
            lineOutcome[i] = 0;
        }

        shct = gm_calloc<uint8_t>(SHCT_SIZE);
        for (uint32_t i = 0; i < SHCT_SIZE; i++) {
            shct[i] = 1;    // recommended initial value for SCHT from the paper 
        }
    }

    ~SHiPReplPolicy() {
        // Frees all allocated arrays to avoid leaks
        gm_free(rrpv);
        gm_free(lineSig);
        gm_free(lineOutcome);
        gm_free(shct);
    }

    // -------------------------------------------------------------------
    // update(): called on ALL accesses including immediately after replaced()
    // -------------------------------------------------------------------
    void update(uint32_t id, const MemReq* req) override {

        // ignore update() for freshly inserted line
        if (id == lastReplaced) {
            lastReplaced = UINT32_MAX;
            return;
        }

        // REAL hit
        if (hitPromote) {
            rrpv[id] = 0;
        }

        // If this line had no signature, skip SHCT training
        if (lineSig[id] == 0) {
            return;
        }

        // If lineOutcome[id] is 0, this is the first hit for this line
        if (lineOutcome[id] == 0) {
            // find the SHCT entry for this line's signature
            uint32_t idx = shctIndex(lineSig[id]);
            // increment that counter by 1, but not beyond SHCT_MAX
            if (shct[idx] < SHCT_MAX) shct[idx]++;
            // mark this line, because it is important for us to give the 
            // 'insertRRPV' value in replaced()
            lineOutcome[id] = 1;
        }
    }

    // -------------------------------------------------------------------
    // replaced(): called AFTER rank() chooses victim
    // -------------------------------------------------------------------
    void replaced(uint32_t id) override {
        // replaced is called when a victim line is chosen and a new line is inserted in its place
        lastReplaced = id; // marks that this line is new so next update should be ignored.

        // signature for the current miss, previously stored in rank.
        uint32_t idx = shctIndex(lastSig);
        // Get the line's hit history based on the previous hits
        uint8_t ctr = shct[idx];

        // Is the line hit before? 
        // if yes, it is important to us. assign rrpvMax - 1 to evict that line slower
        // if no, then the line is not important to us, assign max value to evict faster
        uint32_t insertRRPV = (ctr == 0) ? rpvMax : (rpvMax - 1);

        rrpv[id]        = insertRRPV;
        lineSig[id]     = lastSig; // store the signature for this line
        lineOutcome[id] = 0; // reset its outcome to 0
    }

    // -------------------------------------------------------------------
    // rank(): victim selection
    // -------------------------------------------------------------------
    template <typename C>
    uint32_t rank(const MemReq* req, C cands) {
        // rank is called to find which line to evict?
        // stores the signature of the current miss in lastSig, so that replaced() can see it later
        lastSig = getSignature(req);

        while (true) {
            for (auto it = cands.begin(); it != cands.end(); it.inc()) {
                uint32_t id = *it;

                // If the line is invalid, we choose it immediately as victim
                if (!cc || !cc->isValid(id)) {
                    return id;
                }

                // If line is equal to rpvMax, it is ready to be evicted
                if (rrpv[id] == rpvMax) {
                    // Train the victim, before evicting it!
                    // If this line never had a hit, its signature's SHCT counter is decremented.
                    trainEviction(id);
                    return id;
                }
            }

            // no victim :( → age everyone untill some line hit with rpvMax
            for (auto it = cands.begin(); it != cands.end(); it.inc()) {
                uint32_t id = *it;
                if (rrpv[id] < rpvMax) rrpv[id]++;
            }
        }
    }

    DECL_RANK_BINDINGS;
};

#endif