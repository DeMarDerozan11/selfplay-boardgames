// Exact solver for Dobutsu Shogi by retrograde analysis (own implementation, no third-party code).
//
// Rules (see README): 3x4 board, lion/giraffe/elephant/chick(hen), drops without restrictions,
// mandatory promotion, win by capturing the lion or by "try" (own lion on the far row and not
// capturable at once). Repetition is not modelled: positions that are never resolved are draws.
//
// Position = (board, hands, side to move). Left-right mirrored positions are equivalent, so only
// the canonical one (smaller key) is stored. Only NON-terminal positions are stored; terminal ones
// are recognised on the fly. Values are exact distances in plies:
//   byte 0 = unresolved (draw), 1+2d = player to move wins in d plies, 2+2d = loses in d plies.
//
// Usage: dobutsu_solver solve <out.tb>     build the table (about 0.6 GB)
//        dobutsu_solver children           stdin: keys -> stdout: terminal code + sorted child keys
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef uint64_t u64;
typedef uint8_t u8;
static int BITS = 27;                       // hash table has 2^BITS slots (set with the 3rd argument)
static u64 NSLOT = 1ULL << 27, MASK = NSLOT - 1;
static u64 stored_count = 0;                // number of occupied slots (to detect a full table)
static const u64 HMUL = 0x9E3779B97F4A7C15ULL;

// cell code = type + 8*owner ; types: 1 chick, 2 giraffe, 3 elephant, 4 lion, 5 hen
struct Pos { int8_t b[12]; int8_t hand[2][3]; int side; };

static u64 encode(const Pos& p) {
    u64 k = 0;
    for (int i = 0; i < 12; i++) k |= (u64)p.b[i] << (4 * i);
    for (int o = 0; o < 2; o++) for (int t = 0; t < 3; t++) k |= (u64)p.hand[o][t] << (48 + (o * 3 + t) * 2);
    k |= (u64)p.side << 60;
    return k;
}
static Pos decode(u64 k) {
    Pos p;
    for (int i = 0; i < 12; i++) p.b[i] = (k >> (4 * i)) & 15;
    for (int o = 0; o < 2; o++) for (int t = 0; t < 3; t++) p.hand[o][t] = (k >> (48 + (o * 3 + t) * 2)) & 3;
    p.side = (k >> 60) & 1;
    return p;
}
static u64 mirror(u64 k) {  // swap columns 0 and 2 in every row
    u64 m = k & ~0xFFFFFFFFFFFFULL;
    for (int r = 0; r < 4; r++) {
        u64 a = (k >> (4 * (3 * r))) & 15, b = (k >> (4 * (3 * r + 1))) & 15, c = (k >> (4 * (3 * r + 2))) & 15;
        m |= c << (4 * (3 * r)) | b << (4 * (3 * r + 1)) | a << (4 * (3 * r + 2));
    }
    return m;
}
static u64 canon(u64 k) { u64 m = mirror(k); return m < k ? m : k; }

static const int DR[6][8] = {{0}, {1}, {1, -1, 0, 0}, {1, 1, -1, -1}, {1, 1, 1, 0, 0, -1, -1, -1}, {1, 1, 1, 0, 0, -1}};
static const int DC[6][8] = {{0}, {0}, {0, 0, 1, -1}, {1, -1, 1, -1}, {-1, 0, 1, -1, 1, -1, 0, 1}, {-1, 0, 1, -1, 1, 0}};
static const int ND[6] = {0, 1, 4, 4, 8, 6};

static inline int far_row(int owner) { return owner == 0 ? 3 : 0; }

static int gen(const Pos& p, Pos* out) {
    int n = 0, s = p.side;
    for (int sq = 0; sq < 12; sq++) {
        int code = p.b[sq];
        if (!code || (code >> 3) != s) continue;
        int t = code & 7;
        for (int d = 0; d < ND[t]; d++) {
            int r = sq / 3 + (s ? -DR[t][d] : DR[t][d]), c = sq % 3 + DC[t][d];
            if (r < 0 || r > 3 || c < 0 || c > 2) continue;
            int dst = r * 3 + c, dc = p.b[dst];
            if (dc && (dc >> 3) == s) continue;
            Pos q = p;
            q.b[sq] = 0;
            if (dc) {
                int ct = dc & 7;
                if (ct != 4) { if (ct == 5) ct = 1; q.hand[s][ct - 1]++; }
            }
            int nt = (t == 1 && r == far_row(s)) ? 5 : t;  // mandatory promotion
            q.b[dst] = nt | (s << 3);
            q.side = 1 - s;
            out[n++] = q;
        }
    }
    for (int t = 0; t < 3; t++) {
        if (!p.hand[s][t]) continue;
        for (int sq = 0; sq < 12; sq++) {
            if (p.b[sq]) continue;
            Pos q = p;
            q.hand[s][t]--;
            q.b[sq] = (t + 1) | (s << 3);
            q.side = 1 - s;
            out[n++] = q;
        }
    }
    return n;
}

static bool can_capture(const Pos& p, int target) {  // can the side to move move a piece onto `target`?
    int s = p.side;
    for (int sq = 0; sq < 12; sq++) {
        int code = p.b[sq];
        if (!code || (code >> 3) != s) continue;
        int t = code & 7;
        for (int d = 0; d < ND[t]; d++) {
            int r = sq / 3 + (s ? -DR[t][d] : DR[t][d]), c = sq % 3 + DC[t][d];
            if (r >= 0 && r <= 3 && c >= 0 && c <= 2 && r * 3 + c == target) return true;
        }
    }
    return false;
}

// 0 = not terminal, 1 = side to move has won (d=0), 2 = side to move has lost (d=0)
static int term(const Pos& p) {
    int s = p.side, own = -1, opp = -1;
    for (int i = 0; i < 12; i++) {
        if (p.b[i] == (4 | (s << 3))) own = i;
        if (p.b[i] == (4 | ((1 - s) << 3))) opp = i;
    }
    if (own < 0) return 2;                              // my lion was captured
    if (own / 3 == far_row(s)) return 1;                // my lion stands on the far row and survived
    if (opp < 0) return 1;                              // (cannot happen in play)
    if (opp / 3 == far_row(1 - s) && !can_capture(p, opp)) return 2;  // opponent's try succeeded
    return 0;
}

static u64* keys;
static u8* val;

static inline u64 slot_of(u64 k) { return (k * HMUL) >> (64 - BITS); }  // note: Python lookup must use the same BITS
static long find(u64 k) {
    for (u64 i = slot_of(k);; i = (i + 1) & MASK) {
        if (keys[i] == k) return (long)i;
        if (!keys[i]) return -1;
    }
}
static bool insert(u64 k, u8 v) {  // returns true if newly inserted
    for (u64 i = slot_of(k);; i = (i + 1) & MASK) {
        if (keys[i] == k) return false;
        if (!keys[i]) {
            if (++stored_count > NSLOT - NSLOT / 8) {  // above 87.5% load: linear probing degrades, stop cleanly
                fprintf(stderr, "hash table is full (%llu entries): rerun with a larger size\n", (unsigned long long)stored_count);
                exit(2);
            }
            keys[i] = k; val[i] = v; return true;
        }
    }
}

static Pos initial() {
    Pos p; memset(&p, 0, sizeof p);
    // first player (owner 0) at the bottom: elephant, lion, giraffe; chick in front of the lion
    p.b[0] = 3; p.b[1] = 4; p.b[2] = 2; p.b[4] = 1;
    // second player (owner 1), rotated by 180 degrees
    p.b[11] = 3 | 8; p.b[10] = 4 | 8; p.b[9] = 2 | 8; p.b[7] = 1 | 8;
    p.side = 0;
    return p;
}

static void cmd_children() {
    char line[64];
    while (fgets(line, sizeof line, stdin)) {
        u64 k = strtoull(line, 0, 10);
        Pos p = decode(k), out[128];
        int tv = term(p);
        std::vector<u64> ks;
        if (!tv) { int n = gen(p, out); for (int i = 0; i < n; i++) ks.push_back(encode(out[i])); }
        std::sort(ks.begin(), ks.end());
        ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
        printf("%d", tv);
        for (u64 x : ks) printf(" %llu", (unsigned long long)x);
        printf("\n");
    }
}

static inline int dist_of(u8 c) { return (c & 1) ? (c - 1) / 2 : (c - 2) / 2; }

static void cmd_solve(const char* out) {
    keys = (u64*)calloc(NSLOT, 8);
    val = (u8*)calloc(NSLOT, 1);
    if (!keys || !val) { fprintf(stderr, "out of memory\n"); exit(1); }
    insert(canon(encode(initial())), 1);
    // 1. enumerate all reachable non-terminal positions, level by level (val = level + 1)
    long total = 1, stuck = 0;
    fprintf(stderr, "table: 2^%d slots = %.0f MB\n", BITS, NSLOT * 9 / 1e6);
    for (int L = 0; L < 250; L++) {
        long added = 0;
        for (u64 i = 0; i < NSLOT; i++) {
            if (!keys[i] || val[i] != L + 1) continue;
            Pos p = decode(keys[i]), ch[128];
            int n = gen(p, ch);
            if (n == 0) stuck++;
            for (int j = 0; j < n; j++) {
                if (term(ch[j])) continue;
                if (insert(canon(encode(ch[j])), (u8)(L + 2))) added++;
            }
        }
        total += added;
        if (!added) break;
        fprintf(stderr, "bfs level %d: +%ld (stored %ld)\n", L + 1, added, total);
    }
    fprintf(stderr, "stored canonical non-terminal positions: %ld, positions without legal move: %ld\n", total, stuck);
    memset(val, 0, NSLOT);
    // 2. retrograde sweeps: in sweep k, only values with distance < k are used (exact distances)
    unsigned nth = std::thread::hardware_concurrency();
    for (int k = 1; k < 120; k++) {
        std::atomic<long> resolved(0);
        auto work = [&](u64 lo, u64 hi) {
            long cnt = 0;
            for (u64 i = lo; i < hi; i++) {
                if (!keys[i] || val[i]) continue;
                Pos p = decode(keys[i]), ch[128];
                int n = gen(p, ch);
                int best_loss = 1000, maxd = -1;
                bool all_win = true;
                for (int j = 0; j < n; j++) {
                    int tv = term(ch[j]);
                    int kind = 0, d = 0;  // kind 1: child is won for its mover, 2: lost, 0: unknown
                    if (tv) { kind = tv; d = 0; }
                    else {
                        long s = find(canon(encode(ch[j])));
                        u8 c = s >= 0 ? val[s] : 0;
                        if (c) { d = dist_of(c); if (d < k) kind = (c & 1) ? 1 : 2; }
                    }
                    if (kind == 2) { if (d < best_loss) best_loss = d; }
                    if (kind == 1) { if (d > maxd) maxd = d; } else all_win = false;
                }
                if (best_loss < 1000) { val[i] = (u8)(1 + 2 * (best_loss + 1)); cnt++; }
                else if (all_win && n > 0) { val[i] = (u8)(2 + 2 * (maxd + 1)); cnt++; }
            }
            resolved += cnt;
        };
        std::vector<std::thread> th;
        u64 chunk = NSLOT / nth + 1;
        for (unsigned t = 0; t < nth; t++) th.emplace_back(work, t * chunk, std::min(NSLOT, (t + 1) * chunk));
        for (auto& t : th) t.join();
        fprintf(stderr, "sweep %d: resolved %ld\n", k, (long)resolved);
        if (!resolved) break;
    }
    // 3. statistics (weighted: a non-symmetric canonical position stands for 2 positions)
    long w = 0, l = 0, d = 0, nt = 0;
    for (u64 i = 0; i < NSLOT; i++) {
        if (!keys[i]) continue;
        long wt = mirror(keys[i]) == keys[i] ? 1 : 2;
        nt += wt;
        if (!val[i]) d += wt; else if (val[i] & 1) w += wt; else l += wt;
    }
    printf("non-terminal positions: %ld\nwon for player to move: %ld\ndrawn: %ld\nlost: %ld\n", nt, w, d, l);
    long s0 = find(canon(encode(initial())));
    u8 c = val[s0];
    printf("initial position: %s in %d plies\n", !c ? "draw" : (c & 1) ? "WIN for first player" : "LOSS for first player",
           c ? dist_of(c) : -1);
    FILE* f = fopen(out, "wb");
    if (!f) { perror("open output"); exit(1); }
    fwrite(keys, 8, NSLOT, f);
    fwrite(val, 1, NSLOT, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc >= 4) { BITS = atoi(argv[3]); NSLOT = 1ULL << BITS; MASK = NSLOT - 1; }
    if (argc >= 3 && !strcmp(argv[1], "solve")) cmd_solve(argv[2]);
    else if (argc >= 2 && !strcmp(argv[1], "children")) cmd_children();
    else { fprintf(stderr, "usage: %s solve <out.tb> | children\n", argv[0]); return 1; }
    return 0;
}
