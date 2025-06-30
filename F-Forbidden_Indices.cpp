//████████╗██╗███╗░░██╗  ██╗░░░░░███████╗
//╚══██╔══╝██║████╗░██║  ██║░░░░░██╔════╝
//░░░██║░░░██║██╔██╗██║  ██║░░░░░█████╗░░
//░░░██║░░░██║██║╚████║  ██║░░░░░██╔══╝░░
//░░░██║░░░██║██║░╚███║  ███████╗███████╗
//░░░╚═╝░░░╚═╝╚═╝░░╚══╝  ╚══════╝╚══════╝
//   __________________
//  | ________________ |
//  ||          ____  ||
//  ||   /\    |      ||
//  ||  /__\   |      ||
//  || /    \  |____  ||
//  ||________________||
//  |__________________|
//  \###################\
//   \###################\
//    \        ____       \
//     \_______\___\_______\
// An AC a day keeps the doctor away.

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <cassert>
#include <complex>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <bitset>
#include <iomanip>
#include <functional>
#include <numeric>
#include <stack>
#include <array>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define ull unsigned long long
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define vc vt<char> 
#define vvc vt<vc>
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vs vt<string>
#define vvs vt<vs>
#define vb vt<bool>
#define vvb vt<vb>
#define vvpii vt<vpii>
#define vd vt<db>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define ff first
#define ss second
#define sv string_view
#define MP make_pair
#define MT make_tuple
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define SORTED(x) is_sorted(all(x))
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128

//SGT DEFINE
#define lc i * 2 + 1
#define rc i * 2 + 2
#define lp lc, left, middle
#define rp rc, middle + 1, right
#define entireTree 0, 0, n - 1
#define midPoint left + (right - left) / 2
#define pushDown push(i, left, right)
#define iter int i, int left, int right

#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)

struct custom {
    static const uint64_t C = 0x9e3779b97f4a7c15; const uint32_t RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const { return __builtin_bswap64((x ^ RANDOM) * C); }
    size_t operator()(const std::string& s) const { size_t hash = std::hash<std::string>{}(s); return hash ^ RANDOM; } };
template <class K, class V> using umap = std::unordered_map<K, V, custom>; template <class K> using uset = std::unordered_set<K, custom>;
template<class T> using max_heap = priority_queue<T>;
template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
    
template<typename T, size_t N>
istream& operator>>(istream& is, array<T, N>& arr) {
    for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is;
}

template<typename T, size_t N>
istream& operator>>(istream& is, vector<array<T, N>>& vec) {
    for (auto &arr : vec) { is >> arr; } return is;
}

inline std::ostream& operator<<(std::ostream& os, i128 x) {
    if(x == 0) { os << '0'; return os; } if(x < 0) { os << '-'; x = -x; }
    string s; while (x > 0) { int digit = int(x % 10); s.pb(char('0' + digit)); x /= 10; }
    rev(s); os << s; return os;
}
    
template <typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) {    return in >> input.ff >> input.ss; }
    
template <typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }

template<class T>
void output_vector(vt<T>& a, int off_set = 0) {
    int n = a.size();
    for(int i = off_set; i < n; i++) {
        cout << a[i] << (i == n - 1 ? '\n' : ' ');
    }
}

template<typename T, typename Compare>
vi closest_left(const vt<T>& a, Compare cmp) {
    int n = a.size(); vi closest(n); iota(all(closest), 0);
    for (int i = 0; i < n; i++) {
        auto& j = closest[i];
        while(j && cmp(a[i], a[j - 1])) j = closest[j - 1];
    }
    return closest;
}

template<typename T, typename Compare> // auto right = closest_right<int>(a, std::less<int>());
vi closest_right(const vt<T>& a, Compare cmp) {
    int n = a.size(); vi closest(n); iota(all(closest), 0);
    for (int i = n - 1; i >= 0; i--) {
        auto& j = closest[i];
        while(j < n - 1 && cmp(a[i], a[j + 1])) j = closest[j + 1];
    }
    return closest;
}

template<typename T, typename V = string>
vt<pair<T, int>> encode(const V& s) {
    vt<pair<T, int>> seg;
    for(auto& ch : s) {
        if(seg.empty() || ch != seg.back().ff) seg.pb({ch, 1});
        else seg.back().ss++;
    }
    return seg;
}

    
template<typename K, typename V>
auto operator<<(std::ostream &o, const std::map<K, V> &m) -> std::ostream& {
    o << "{"; int i = 0;
    for (const auto &[key, value] : m) { if (i++) o << " , "; o << key << " : " << value; }
    return o << "}";
}

#ifdef LOCAL
#define debug(x...) debug_out(#x, x)
void debug_out(const char* names) { std::cerr << std::endl; }
template <typename T, typename... Args>
void debug_out(const char* names, T value, Args... args) {
    const char* comma = strchr(names, ',');
    std::cerr << "[" << (comma ? std::string(names, comma) : names) << " = " << value << "]";
    if (sizeof...(args)) { std::cerr << ", "; debug_out(comma + 1, args...); }   
    else { std::cerr << std::endl; }
}
template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& o, const std::pair<T1, T2>& p) { return o << "{" << p.ff << " , " << p.ss << "}"; }
auto operator<<(auto &o, const auto &x) -> decltype(end(x), o) {
    o << "{"; int i = 0; for (const auto &e : x) { if (i++) o << " , "; o << e; } return o << "}";
} // remove for leetcode
#include <sys/resource.h>
#include <sys/time.h>
void printMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double memoryMB = usage.ru_maxrss / 1024.0;
    cerr << "Memory usage: " << memoryMB << " MB" << "\n";
}

#define startClock clock_t tStart = clock();
#define endClock std::cout << std::fixed << std::setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << " seconds" << std::endl;
#else
#define debug(...)
#define startClock
#define endClock

#endif
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

#define eps 1e-9
#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
const static ll INF = 1LL << 62;
const static int inf = 1e9 + 100;
const static int MK = 20;
const static int MX = 1e5 + 5;
ll gcd(ll a, ll b) { while (b != 0) { ll temp = b; b = a % b; a = temp; } return a; }
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
ll floor(ll a, ll b) { if(b < 0) a = -a, b = -b; if (a >= 0) return a / b; return a / b - (a % b ? 1 : 0); }
ll ceil(ll a, ll b) { if (b < 0) a = -a, b = -b; if (a >= 0) return (a + b - 1) / b; return a / b; }
int pct(ll x) { return __builtin_popcountll(x); }
ll have_bit(ll x, int b) { return x & (1LL << b); }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const vc dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) {return n - sum_even_series(n);} // sum of first n odd number is n ^ 2
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^1 + p^2 + p^3 + ... + p^k
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
bool is_vowel(char c) {return c == 'a' || c == 'e' || c == 'u' || c == 'o' || c == 'i';}

class suffix_array {
    public:
    template <typename T, typename F = function<bool(const T&, const T&)>> // only handle max, min
        struct linear_rmq {
            vt<T> values;
            F compare;
            vi head;
            vt<array<unsigned,2>> masks;

            linear_rmq() {}

            linear_rmq(const vt<T>& arr, F cmp = F{})
                : values(arr), compare(cmp),
                head(arr.size()+1),
                masks(arr.size())
                {
                    vi monoStack{-1};
                    int n = arr.size();
                    for (int i = 0; i <= n; i++) {
                        int last = -1;
                        while (monoStack.back() != -1 &&
                                (i == n || !compare(values[monoStack.back()], values[i])))
                        {
                            if (last != -1) head[last] = monoStack.back();
                            unsigned diffBit = __bit_floor(unsigned(monoStack.end()[-2] + 1) ^ i);
                            masks[monoStack.back()][0] = last = (i & -diffBit);
                            monoStack.pop_back();
                            masks[monoStack.back() + 1][1] |= diffBit;
                        }
                        if (last != -1) head[last] = i;
                        monoStack.pb(i);
                    }
                    for (int i = 1; i < n; i++) {
                        masks[i][1] = (masks[i][1] | masks[i-1][1])
                            & -(masks[i][0] & -masks[i][0]);
                    }
                }

            T query(int L, int R) const {
                unsigned common = masks[L][1] & masks[R][1]
                    & -__bit_floor((masks[L][0] ^ masks[R][0]) | 1);
                unsigned k = masks[L][1] ^ common;
                if (k) {
                    k = __bit_floor(k);
                    L = head[(masks[L][0] & -k) | k];
                }
                k = masks[R][1] ^ common;
                if (k) {
                    k = __bit_floor(k);
                    R = head[(masks[R][0] & -k) | k];
                }
                return compare(values[L], values[R]) ? values[L] : values[R];
            }
        };
    string s;
    int n;
    vi sa, pos, lcp;
    ll distinct_substring;
    linear_rmq<int> rmq;
    suffix_array() {}

    suffix_array(const string& s) {
        this->s = s;
        distinct_substring = 0;
        n = s.size();
        sa.rsz(n), pos.rsz(n), lcp.rsz(n);
        init();
        build_lcp();
        rmq = linear_rmq<int>(lcp, [](const int& a, const int& b) {return a < b;});
    }

    int get_lcp(int i, int j) {
        if(i == j) return s.size() - i;
        i = pos[i], j = pos[j];
        if(i > j) swap(i, j);
        return rmq.query(i, j - 1);
    }

    void sorted_substring(vpii& S) {
        // https://codeforces.com/edu/course/2/lesson/2/5/practice/status
        sort(all(S), [&](const pii &a, const pii& b) {
                    auto& [l1, r1] = a;
                    auto& [l2, r2] = b;
                    int len1 = r1 - l1 + 1;
                    int len2 = r2 - l2 + 1;
                    int common = get_lcp(l1, l2);
                    debug(a, b, common);
                    if(common >= min(len1, len2)) {
                        if(len1 != len2) return len1 < len2;
                        return l1 < l2;
                    }
                    return s[l1 + common] < s[l2 + common];
                });
    }


    void init() {
        vi r(n), tmp(n), sa2(n), cnt(max(256, n) + 1);

        for(int i = 0; i < n; i++) {
            sa[i] = i;
            r[i] = s[i];
        }

        for(int k = 1; k < n; k <<= 1) {
            fill(all(cnt), 0);
            for(int i = 0; i < n; i++) {
                int key = (i + k < n ? r[i + k] + 1 : 0);
                cnt[key]++;
            }
            for(int i = 1; i < (int)cnt.size(); i++) cnt[i] += cnt[i - 1];
            for(int i = n - 1; i >= 0; i--) {
                int j = sa[i];
                int key = (j + k < n ? r[j + k] + 1 : 0);
                sa2[--cnt[key]] = j;
            }

            fill(all(cnt), 0);
            for(int i = 0; i < n; i++) {
                cnt[r[i] + 1]++;
            }
            for(int i = 1; i < (int)cnt.size(); i++) cnt[i] += cnt[i - 1];
            for(int i = n - 1; i >= 0; i--) {
                int j = sa2[i];
                sa[--cnt[r[j] + 1]] = j;
            }

            tmp[sa[0]] = 0;
            for(int i = 1; i < n; i++) {
                auto [a1, b1] = MP(r[sa[i - 1]], sa[i - 1] + k < n ? r[sa[i - 1] + k] : -1);
                auto [a2, b2] = MP(r[sa[i]], sa[i] + k < n ? r[sa[i] + k] : -1);
                tmp[sa[i]] = tmp[sa[i - 1]] + (a1 != a2 || b1 != b2);
            }
            r = tmp;
            if(r[sa[n - 1]] == n - 1) break;  
        }
        for(int i = 0; i < n; i++) {
            pos[sa[i]] = i;
        }
    }

    void build_lcp() {
        for(int i = 0, k = 0; i < n; i++) {
            if(pos[i] == n - 1) continue;
            int j = sa[pos[i] + 1];
            while(s[i + k] == s[j + k]) k++;
            lcp[pos[i]] = k;
            if(k) k--;
        }
        distinct_substring = (ll)n * (n + 1) / 2 - sum(lcp);
    }

    int check(const string& x, int m) {
        int j = sa[m];
        int L = min((int)x.size(), n - j);
        for(int i = 0; i < L; i++) {
            if(s[j + i] < x[i]) return -1;
            if(s[j + i] > x[i]) return  1;
        }
        if((int)x.size() == L) return 0;
        return -1;
    }
     
    pii get_bound(const string& x) {
        int l = 0, r = n - 1, first = -1;
        while(l <= r) {
            int m = (l + r) >> 1;
            int v = check(x, m);
            if(v >= 0) {
                if(v == 0) first = m;
                r = m - 1;
            } else {
                l = m + 1;
            }
        }
        if(first == -1) return {-1, -1};
        l = first; 
        r = n - 1;
        int last = first;
        while(l <= r) {
            int m = (l + r) >> 1;
            int v = check(x, m);
            if(v <= 0) {
                if(v == 0) last = m;
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return {first, last};
    }

    int count(const string& x) {
        if(x.size() > n) return 0;
        auto [l, r] = get_bound(x);
        return l == -1 ? 0 : r - l + 1;
    }

    string lcs(const string& s, const string& t) {
        string combined = s + '$' + t;
        suffix_array sa_combined(combined);
        int max_lcp = 0, start_pos = 0;
        int split = s.size();
        for(int i = 1; i < sa_combined.n; i++) {
            int suffix1 = sa_combined.sa[i - 1];
            int suffix2 = sa_combined.sa[i];
            bool in_s1 = suffix1 < split;
            bool in_t1 = suffix2 > split;
            bool in_s2 = suffix2 < split;
            bool in_t2 = suffix1 > split;
            if((in_s1 && in_t1) || (in_s2 && in_t2)) {
                int len = sa_combined.lcp[i - 1];
                if(len > max_lcp) {
                    max_lcp = len;
                    start_pos = sa_combined.sa[i];
                }
            }
        }
        return combined.substr(start_pos, max_lcp);
    }

    string kth_distinct(ll k) {
        if(k > (ll)n * (n + 1) / 2) return "";
        ll prev = 0, curr = 0;
        for(int i = 0; i < n; i++) {
            if(curr + (n - sa[i]) - prev >= k) {
                string ans = s.substr(sa[i], prev);
                while(curr < k) {
                    ans += s[sa[i] + prev++];
                    curr++;
                }
                return ans;
            }
            curr += (n - sa[i]) - prev;
            prev = lcp[i];
        }
        return "";
    }

    string lcs(vs& a) {
        int K = a.size();
        if(K == 0) return "";
        if(K == 1) return a[0];

        int total = 0;
        for(auto &s : a) total += s.size() + 1;
        string T; 
        T.reserve(total);
        vi owner;
        owner.reserve(total);
        for(int i = 0; i < K; i++) {
            for(char& c : a[i]) {
                T.pb(c);
                owner.pb(i);
            }
            T.pb(char(1 + i));
            owner.pb(-1);
        }

        suffix_array sa2(T);
        int N2 = sa2.n;

        vi freq(K);
        int have = 0, left = 0;
        int best = 0, bestPos = 0;
        deque<pii> dq;

        for(int right = 0; right < N2; right++) {
            int id = owner[sa2.sa[right]];
            if(id >= 0 && ++freq[id] == 1) have++;

            if(right > 0) {
                int idx = right - 1;
                int v = sa2.lcp[idx];
                while(!dq.empty() && dq.back().ss >= v) dq.pop_back();
                dq.emplace_back(idx, v);
            }

            while(have == K) {
                while(!dq.empty() && dq.front().ff < left) dq.pop_front();
                if(left < right && !dq.empty() && dq.front().ss > best) {
                    best = dq.front().ss;
                    bestPos = sa2.sa[dq.front().ff];
                }
                int idL = owner[sa2.sa[left]];
                if(idL >= 0 && --freq[idL] == 0) have--;
                left++;
            }
        }
        return best > 0 ? T.substr(bestPos, best) : string();
    }

    vi lcp_vector(const string& s, const string& t) { // return a vector for each i in t represents the lcp in s
        int n = s.size(), m = t.size();
        const int N = n + m + 1;
        suffix_array S(s + '#' + t);
        vi prev(N, -1), next(N, -1);
        for(int i = 0; i < N; i++) {
            if(i) prev[i] = prev[i - 1];
            int p = S.sa[i];
            if(p < n) prev[i] = i;
        }
        for(int i = N - 1; i >= 0; i--) {
            if(i < N - 1) next[i] = next[i + 1];
            int p = S.sa[i];
            if(p < n) next[i] = i;
        }
        vi A(m);
        for(int i = n + 1; i < N; i++) {
            int p = S.pos[i];
            int mx = 0;
            if(prev[p] != -1) {
                mx = max(mx, S.get_lcp(i, S.sa[prev[p]]));
            }
            if(next[p] != -1) {
                mx = max(mx, S.get_lcp(i, S.sa[next[p]]));
            }
            A[i - (n + 1)] = mx;
        }
        return A;
    }
};

void solve() {
    ll n; cin >> n;
    string s, t; cin >> s >> t;
    rev(s), rev(t);
    ll res = 0;
    suffix_array S(s);
    vi b(n);
    for(int i = 0; i < n; i++) {
        int p = S.sa[i];
        b[i] = t[p] - '0';
        if(t[i] == '0') {
            res = max(res, n - i);
        }
    }
    vi pre(n + 1);
    for(int i = 0; i < n; i++) {
        pre[i + 1] = pre[i] + int(b[i] == 0);
    }
    auto left = closest_left(S.lcp, less_equal<int>());
    auto right = closest_right(S.lcp, less_equal<int>());
    debug(left, right, S.lcp, S.sa, S.pos);
    for(int i = 0; i < n; i++) {
        int L = left[i], R = right[i];
        res = max(res, (ll)S.lcp[i] * (pre[R + 1] - pre[L] + (R + 1 < n && b[R + 1] == 0)));
    }
    cout << res << '\n';
    
}

signed main() {
    // careful for overflow, check for long long, use unsigned long long for random generator
    // when mle, look if problem require read in file, typically old problems
    IOS;
    startClock
    //generatePrime();

    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }

    endClock
    #ifdef LOCAL
      printMemoryUsage();
    #endif

    return 0;
}

//███████████████████████████████████████████████████████████████████████████████████████████████████████
//█░░░░░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░███░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░░░█
//█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░░░░░░░░░██░░▄▀░░█░░▄▀▄▀▄▀▄▀░░░░█░░▄▀▄▀▄▀░░█░░▄▀░░░░░░░░░░██░░▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█
//█░░▄▀░░░░░░░░░░█░░▄▀▄▀▄▀▄▀▄▀░░██░░▄▀░░█░░▄▀░░░░▄▀▄▀░░█░░░░▄▀░░░░█░░▄▀▄▀▄▀▄▀▄▀░░██░░▄▀░░█░░▄▀░░░░░░░░░░█
//█░░▄▀░░█████████░░▄▀░░░░░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░░░░░▄▀░░██░░▄▀░░█░░▄▀░░█████████
//█░░▄▀░░░░░░░░░░█░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░█████████
//█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░░░░░█
//█░░▄▀░░░░░░░░░░█░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█
//█░░▄▀░░█████████░░▄▀░░██░░▄▀░░░░░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░░░░░▄▀░░█░░▄▀░░██░░▄▀░░█
//█░░▄▀░░░░░░░░░░█░░▄▀░░██░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░░░▄▀▄▀░░█░░░░▄▀░░░░█░░▄▀░░██░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░░░░░▄▀░░█
//█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░██░░░░░░░░░░▄▀░░█░░▄▀▄▀▄▀▄▀░░░░█░░▄▀▄▀▄▀░░█░░▄▀░░██░░░░░░░░░░▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█
//█░░░░░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░███░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░░░█
//███████████████████████████████████████████████████████████████████████████████████████████████████████
