#ifndef DRRIP_REPL_H_
#define DRRIP_REPL_H_

#include "repl_policies.h"

// Dynamic RRIP (DRRIP)
class DRRIPReplPolicy : public ReplPolicy {
private:
    uint8_t* rrpv;        // RRPV per line
    bool*    justInserted; // Marks lines that were just inserted (to ignore first update)
    uint32_t numLines;
    uint32_t numWays;
    uint32_t numSets;
    uint32_t rpvMax;

    // BRRIP state
    uint32_t brripCounter;  // For 1/32 vs 31/32 probability

    // PSEL state (saturating counter for policy selection)
    int32_t  psel;
    uint32_t pselMax;
    uint32_t pselThreshold;

    inline uint32_t getSetId(uint32_t id) const {
        return id / numWays;
    }

    // Leader set selection (simple modulo-based Set Dueling Monitor)
    inline bool isSRRIPLeader(uint32_t setId) const {
        return (setId % 32) == 0;  // e.g., every 32nd set
    }

    inline bool isBRRIPLeader(uint32_t setId) const {
        return (setId % 32) == 1;  // e.g., next set
    }

    // Followers choose which policy to use based on PSEL.
    // Low PSEL  -> SRRIP is better -> use SRRIP
    // High PSEL -> BRRIP is better   -> use BRRIP
    inline bool followersUseSRRIP() const {
        return (uint32_t)psel < pselThreshold;
    }

    // SRRIP-HP insertion: insert as "long but not distant" (rpvMax-1)
    inline void insertAsSRRIP(uint32_t id) {
        rrpv[id] = (rpvMax > 0) ? (rpvMax - 1) : 0;
    }

    // BRRIP insertion:
    //  - 31/32: insert like SRRIP-HP (near)
    //  - 1/32 : insert as distant (rpvMax)
    inline void insertAsBRRIP(uint32_t id) {
        if ((brripCounter++ & 0x1F) == 0) {
            // Rare case (1/32): distant re-reference
            rrpv[id] = rpvMax;
        } else {
            // Common case (31/32): near re-reference (SRRIP-HP style)
            insertAsSRRIP(id);
        }
    }

public:
    DRRIPReplPolicy(uint32_t _numLines, uint32_t _numWays, uint32_t _rpvMax = 3)
        : numLines(_numLines),
          numWays(_numWays),
          rpvMax(_rpvMax)
    {
        numSets = numLines / numWays;

        rrpv = gm_calloc<uint8_t>(numLines);
        justInserted = gm_calloc<bool>(numLines);

        for (uint32_t i = 0; i < numLines; i++) {
            rrpv[i] = rpvMax;      // start life as distant
            justInserted[i] = false;
        }

        // PSEL: 10-bit saturating counter by default
        pselMax       = (1u << 10) - 1;
        pselThreshold = pselMax / 2;
        psel          = (int32_t)pselThreshold;

        brripCounter = 0;
    }

    ~DRRIPReplPolicy() {
        gm_free(rrpv);
        gm_free(justInserted);
    }

    // update(): called after every access
    void update(uint32_t id, const MemReq* req) override {
        // If this line was just inserted due to a miss, ignore the first update()
        // so we do not overwrite the insertion RRPV.
        if (justInserted[id]) {
            justInserted[id] = false;
            return;
        }

        // Real hit: promote to "imminent reuse"
        rrpv[id] = 0;
    }

    // replaced(): called when a line is selected as victim and is being replaced
    void replaced(uint32_t id) override {
        uint32_t setId = getSetId(id);

        if (isSRRIPLeader(setId)) {
            // Miss in SRRIP leader set => SRRIP doing worse here => move PSEL toward BRRIP
            if ((uint32_t)psel < pselMax) {
                psel++;
            }
            insertAsSRRIP(id);
        } else if (isBRRIPLeader(setId)) {
            // Miss in BRRIP leader set => BRRIP doing worse here => move PSEL toward SRRIP
            if (psel > 0) {
                psel--;
            }
            insertAsBRRIP(id);
        } else {
            // Follower set: choose policy based on PSEL
            if (followersUseSRRIP()) {
                insertAsSRRIP(id);
            } else {
                insertAsBRRIP(id);
            }
        }

        // Mark that this line was just inserted so the next update() (called immediately
        // after replace() on a miss) does not overwrite the insertion RRPV.
        justInserted[id] = true;
    }

    // rank(): standard RRIP victim selection
    template <typename C>
    inline uint32_t rank(const MemReq* req, C cands) {
        while (true) {
            // First pass: look for a line with RRPV == rpvMax
            for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                uint32_t id = *ci;
                
                // If the line is invalid, we choose it immediately as victim
                if (!cc || !cc->isValid(id)) 
                    return id;   
                
                if (rrpv[id] >= rpvMax) {
                    return id;
                }
            }

            // Second pass: age all candidates (saturating increment)
            for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                uint32_t id = *ci;
                if (rrpv[id] < rpvMax) {
                    rrpv[id]++;
                }
            }
        }
    }

    DECL_RANK_BINDINGS;
};

#endif // DRRIP_REPL_H_