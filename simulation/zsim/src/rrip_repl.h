#ifndef RRIP_REPL_H_
#define RRIP_REPL_H_

#include "repl_policies.h"

// Static RRIP
class SRRIPReplPolicy : public ReplPolicy {
    protected:

    uint32_t* rrpv_array;   // RRPV values for each cache line
    bool*    justInserted; // Marks lines that were just inserted (to ignore first update)
    uint32_t numLines;      // Number of cache lines
    uint32_t rpvMax;        // Maximum RRPV (e.g., 3 for 2-bit SRRIP)

public:
    // Constructor
    SRRIPReplPolicy(uint32_t _numLines, uint32_t _rpvMax)
        : numLines(_numLines), rpvMax(_rpvMax)
    {
        rrpv_array = gm_calloc<uint32_t>(numLines);
        justInserted = gm_calloc<bool>(numLines);

        for (uint32_t i = 0; i < numLines; i++) {
            rrpv_array[i] = rpvMax;  // Start as "distantly re-referenced"
            justInserted[i] = false;
        }
    }

    // Destructor
    ~SRRIPReplPolicy() {
        gm_free(rrpv_array);
    }

    // Rank candidates for replacement
    virtual uint32_t rankCands(const MemReq* req, SetAssocCands cands) override {
        return rankCandsTemplate(req, cands);
    }

    virtual uint32_t rankCands(const MemReq* req, ZCands cands) override {
        return rankCandsTemplate(req, cands);
    }

    // Return which candidate will be evicted
    template <typename C>
    uint32_t rankCandsTemplate(const MemReq* req, C cands) {
        ///Always returns the first item with rrpv value = rpvMax
        while (true) {
            // Look for any line that has reached the max RRPV
            for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                uint32_t id = *ci;
                
                // If the line is invalid, we choose it immediately as victim
                if (!cc || !cc->isValid(id)) 
                    return id;
                
                if (rrpv_array[id] == rpvMax)
                    return id;  // Found victim line
            }

            // If none are at max, age all lines
            for (auto ci = cands.begin(); ci != cands.end(); ci.inc()) {
                uint32_t id = *ci;
                if (rrpv_array[id] < rpvMax)
                    rrpv_array[id]++;
            }
        }
    }

    // Called when a line is accessed (i.e., hit)
    void update(uint32_t id, const MemReq* req) {
        // If this line was just inserted due to a miss, ignore the first update()
        // so we do not overwrite the insertion RRPV.
        if (justInserted[id]) {
            justInserted[id] = false;
            return;
        }        
        // Hit, so set RRPV to 0 (recently used)
        rrpv_array[id] = 0;
    }

    // Called when a line is replaced with a new block
    void replaced(uint32_t id) {
        // Insert with RRPV = rpvMax - 1 
        rrpv_array[id] = rpvMax - 1;

        // Mark that this line was just inserted so the next update() (called immediately
        // after replace() on a miss) does not overwrite the insertion RRPV.
        justInserted[id] = true;
    }
};
#endif // RRIP_REPL_H_


