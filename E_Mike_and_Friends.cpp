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
ll have_bit(ll x, int b) { return (x >> b) & 1; }
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

#ifndef ATCODER_STRING_HPP
#define ATCODER_STRING_HPP 1

#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>
#include <vector>

namespace atcoder {

namespace internal {

std::vector<int> sa_naive(const std::vector<int>& s) {
    int n = int(s.size());
    std::vector<int> sa(n);
    std::iota(sa.begin(), sa.end(), 0);
    std::sort(sa.begin(), sa.end(), [&](int l, int r) {
        if (l == r) return false;
        while (l < n && r < n) {
            if (s[l] != s[r]) return s[l] < s[r];
            l++;
            r++;
        }
        return l == n;
    });
    return sa;
}

std::vector<int> sa_doubling(const std::vector<int>& s) {
    int n = int(s.size());
    std::vector<int> sa(n), rnk = s, tmp(n);
    std::iota(sa.begin(), sa.end(), 0);
    for (int k = 1; k < n; k *= 2) {
        auto cmp = [&](int x, int y) {
            if (rnk[x] != rnk[y]) return rnk[x] < rnk[y];
            int rx = x + k < n ? rnk[x + k] : -1;
            int ry = y + k < n ? rnk[y + k] : -1;
            return rx < ry;
        };
        std::sort(sa.begin(), sa.end(), cmp);
        tmp[sa[0]] = 0;
        for (int i = 1; i < n; i++) {
            tmp[sa[i]] = tmp[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
        }
        std::swap(tmp, rnk);
    }
    return sa;
}

// SA-IS, linear-time suffix array construction
// Reference:
// G. Nong, S. Zhang, and W. H. Chan,
// Two Efficient Algorithms for Linear Time Suffix Array Construction
template <int THRESHOLD_NAIVE = 10, int THRESHOLD_DOUBLING = 40>
std::vector<int> sa_is(const std::vector<int>& s, int upper) {
    int n = int(s.size());
    if (n == 0) return {};
    if (n == 1) return {0};
    if (n == 2) {
        if (s[0] < s[1]) {
            return {0, 1};
        } else {
            return {1, 0};
        }
    }
    if (n < THRESHOLD_NAIVE) {
        return sa_naive(s);
    }
    if (n < THRESHOLD_DOUBLING) {
        return sa_doubling(s);
    }

    std::vector<int> sa(n);
    std::vector<bool> ls(n);
    for (int i = n - 2; i >= 0; i--) {
        ls[i] = (s[i] == s[i + 1]) ? ls[i + 1] : (s[i] < s[i + 1]);
    }
    std::vector<int> sum_l(upper + 1), sum_s(upper + 1);
    for (int i = 0; i < n; i++) {
        if (!ls[i]) {
            sum_s[s[i]]++;
        } else {
            sum_l[s[i] + 1]++;
        }
    }
    for (int i = 0; i <= upper; i++) {
        sum_s[i] += sum_l[i];
        if (i < upper) sum_l[i + 1] += sum_s[i];
    }

    auto induce = [&](const std::vector<int>& lms) {
        std::fill(sa.begin(), sa.end(), -1);
        std::vector<int> buf(upper + 1);
        std::copy(sum_s.begin(), sum_s.end(), buf.begin());
        for (auto d : lms) {
            if (d == n) continue;
            sa[buf[s[d]]++] = d;
        }
        std::copy(sum_l.begin(), sum_l.end(), buf.begin());
        sa[buf[s[n - 1]]++] = n - 1;
        for (int i = 0; i < n; i++) {
            int v = sa[i];
            if (v >= 1 && !ls[v - 1]) {
                sa[buf[s[v - 1]]++] = v - 1;
            }
        }
        std::copy(sum_l.begin(), sum_l.end(), buf.begin());
        for (int i = n - 1; i >= 0; i--) {
            int v = sa[i];
            if (v >= 1 && ls[v - 1]) {
                sa[--buf[s[v - 1] + 1]] = v - 1;
            }
        }
    };

    std::vector<int> lms_map(n + 1, -1);
    int m = 0;
    for (int i = 1; i < n; i++) {
        if (!ls[i - 1] && ls[i]) {
            lms_map[i] = m++;
        }
    }
    std::vector<int> lms;
    lms.reserve(m);
    for (int i = 1; i < n; i++) {
        if (!ls[i - 1] && ls[i]) {
            lms.push_back(i);
        }
    }

    induce(lms);

    if (m) {
        std::vector<int> sorted_lms;
        sorted_lms.reserve(m);
        for (int v : sa) {
            if (lms_map[v] != -1) sorted_lms.push_back(v);
        }
        std::vector<int> rec_s(m);
        int rec_upper = 0;
        rec_s[lms_map[sorted_lms[0]]] = 0;
        for (int i = 1; i < m; i++) {
            int l = sorted_lms[i - 1], r = sorted_lms[i];
            int end_l = (lms_map[l] + 1 < m) ? lms[lms_map[l] + 1] : n;
            int end_r = (lms_map[r] + 1 < m) ? lms[lms_map[r] + 1] : n;
            bool same = true;
            if (end_l - l != end_r - r) {
                same = false;
            } else {
                while (l < end_l) {
                    if (s[l] != s[r]) {
                        break;
                    }
                    l++;
                    r++;
                }
                if (l == n || s[l] != s[r]) same = false;
            }
            if (!same) rec_upper++;
            rec_s[lms_map[sorted_lms[i]]] = rec_upper;
        }

        auto rec_sa =
            sa_is<THRESHOLD_NAIVE, THRESHOLD_DOUBLING>(rec_s, rec_upper);

        for (int i = 0; i < m; i++) {
            sorted_lms[i] = lms[rec_sa[i]];
        }
        induce(sorted_lms);
    }
    return sa;
}

}  // namespace internal

std::vector<int> suffix_array(const std::vector<int>& s, int upper) {
    assert(0 <= upper);
    for (int d : s) {
        assert(0 <= d && d <= upper);
    }
    auto sa = internal::sa_is(s, upper);
    return sa;
}

template <class T> std::vector<int> suffix_array(const std::vector<T>& s) {
    int n = int(s.size());
    std::vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int l, int r) { return s[l] < s[r]; });
    std::vector<int> s2(n);
    int now = 0;
    for (int i = 0; i < n; i++) {
        if (i && s[idx[i - 1]] != s[idx[i]]) now++;
        s2[idx[i]] = now;
    }
    return internal::sa_is(s2, now);
}

std::vector<int> suffix_array(const std::string& s) {
    int n = int(s.size());
    std::vector<int> s2(n);
    for (int i = 0; i < n; i++) {
        s2[i] = s[i];
    }
    return internal::sa_is(s2, 255);
}

// Reference:
// T. Kasai, G. Lee, H. Arimura, S. Arikawa, and K. Park,
// Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its
// Applications
template <class T>
std::vector<int> lcp_array(const std::vector<T>& s,
                           const std::vector<int>& sa) {
    assert(s.size() == sa.size());
    int n = int(s.size());
    assert(n >= 1);
    std::vector<int> rnk(n);
    for (int i = 0; i < n; i++) {
        assert(0 <= sa[i] && sa[i] < n);
        rnk[sa[i]] = i;
    }
    std::vector<int> lcp(n - 1);
    int h = 0;
    for (int i = 0; i < n; i++) {
        if (h > 0) h--;
        if (rnk[i] == 0) continue;
        int j = sa[rnk[i] - 1];
        for (; j + h < n && i + h < n; h++) {
            if (s[j + h] != s[i + h]) break;
        }
        lcp[rnk[i] - 1] = h;
    }
    return lcp;
}

std::vector<int> lcp_array(const std::string& s, const std::vector<int>& sa) {
    int n = int(s.size());
    std::vector<int> s2(n);
    for (int i = 0; i < n; i++) {
        s2[i] = s[i];
    }
    return lcp_array(s2, sa);
}

// Reference:
// D. Gusfield,
// Algorithms on Strings, Trees, and Sequences: Computer Science and
// Computational Biology
template <class T> std::vector<int> z_algorithm(const std::vector<T>& s) {
    int n = int(s.size());
    if (n == 0) return {};
    std::vector<int> z(n);
    z[0] = 0;
    for (int i = 1, j = 0; i < n; i++) {
        int& k = z[i];
        k = (j + z[j] <= i) ? 0 : std::min(j + z[j] - i, z[i - j]);
        while (i + k < n && s[k] == s[i + k]) k++;
        if (j + z[j] < i + z[i]) j = i;
    }
    z[0] = n;
    return z;
}

std::vector<int> z_algorithm(const std::string& s) {
    int n = int(s.size());
    std::vector<int> s2(n);
    for (int i = 0; i < n; i++) {
        s2[i] = s[i];
    }
    return z_algorithm(s2);
}

}  // namespace atcoder

#endif  // ATCODER_STRING_HPP

class suffix_array { // O(n) suffix_array
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
        sa = atcoder::suffix_array(s);
        lcp = atcoder::lcp_array(s, sa);
        while(lcp.size() < n) lcp.pb(0);
        pos.rsz(n);
        for(int i = 0; i < n; i++) {
            pos[sa[i]] = i;
        }
        distinct_substring = (ll)n * (n + 1) / 2 - sum(lcp);
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

struct wavelet_psgt {
    private:
    struct Node {
        int cnt;
        ll sm;
        Node(int cnt = 0, ll sm = 0) : cnt(cnt), sm(sm) {}
    };
    Node merge(const Node& a, const Node& b) {
        return {a.cnt + b.cnt, a.sm + b.sm};
    }
    Node subtract(const Node& a, const Node& b) {
        return {a.cnt - b.cnt, a.sm - b.sm};
    }
    int n;
    vt<Node> root;
    vi t;
    vpii child;
    vi a;
    int new_node() { root.pb(Node(0, 0)); child.pb({0, 0}); return root.size() - 1; }
    int get_id(ll x) { return int(ub(all(a), x) - begin(a)) - 1; }
    public:
    wavelet_psgt() {}

    wavelet_psgt(const vi& arr) : a(arr) {
        t.rsz(arr.size());
        new_node(); 
        srtU(a);
        n = a.size();
        for(int i = 0, prev = 0; i < (int)arr.size(); i++) {
            t[i] = new_node();
            update(t[i], prev, get_id(arr[i]), Node(1, arr[i]), 0, n - 1);
            prev = t[i];
        }
    }

    void update(int curr, int prev, int id, Node delta, int left, int right) {  
        root[curr] = root[prev];    
        child[curr] = child[prev];
        if(left == right) { 
            root[curr] = merge(root[curr], delta);
            return;
        }
        int middle = midPoint;
        if(id <= middle) child[curr].ff = new_node(), update(child[curr].ff, child[prev].ff, id, delta, left, middle); 
        else child[curr].ss = new_node(), update(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

    int kth(int l, int r, int k) {
        return kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    ll sum_kth(int l, int r, int k) {
        return sum_kth((l == 0 ? 0 : t[l - 1]), t[r], k, 0, n - 1);
    }

    int kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) return -inf;
        if(left == right) return a[left];
        int middle = midPoint;
        int left_cnt = root[child[r].ff].cnt - root[child[l].ff].cnt;
        if(left_cnt >= k) return kth(child[l].ff, child[r].ff, k, left, middle);
        return kth(child[l].ss, child[r].ss, k - left_cnt, middle + 1, right);
    }

    ll sum_kth(int l, int r, int k, int left, int right) {
        if(root[r].cnt - root[l].cnt < k) return -inf;
        if(k <= 0) return 0;
        if(left == right) return (ll)k * a[left];
        int middle = midPoint;
        int left_cnt = root[child[r].ff].cnt - root[child[l].ff].cnt;
        if(left_cnt >= k) return sum_kth(child[l].ff, child[r].ff, k, left, middle); 
        return root[child[r].ff].sm - root[child[l].ff].sm + sum_kth(child[l].ss, child[r].ss, k - left_cnt, middle + 1, right);
    }

    int median(int l, int r) {
        return kth(l, r, (r - l + 2) / 2);
    }

    Node query_leq(int l, int r, int x) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], 0, get_id(x), 0, n - 1);
    }

    Node query_eq(int l, int r, int x) {
        return subtract(query_leq(l, r, x), query_leq(l, r, x - 1));
    }

    Node queries_range(int l, int r, int low, int high) {
        return query((l == 0 ? 0 : t[l - 1]), t[r], get_id(low - 1) + 1, get_id(high), 0, n - 1);
    }

    Node query(int l, int r, int start, int end, int left, int right) {
        if(left > end || right < start || left > right) return Node();
        if(start <= left && right <= end) return subtract(root[r], root[l]);
        int middle = midPoint;
        return merge(query(child[l].ff, child[r].ff, start, end, left, middle), query(child[l].ss, child[r].ss, start, end, middle + 1, right));
    }
	
	ll first_missing_number(int l, int r) { // https://cses.fi/problemset/task/2184/
        ll s = 1;
        return first_missing_number(l == 0 ? 0 : t[l - 1], t[r], 0, n - 1, s);
    }

    ll first_missing_number(ll l, ll r, ll left, ll right, ll &s) {
        if(s < a[left]) return s;
        Node seg = subtract(root[r], root[l]);
        if(a[right] <= s) {
            s += seg.sm;
            return s;
        }
        ll middle = midPoint;
        first_missing_number(child[l].ff, child[r].ff, left, middle, s);
        first_missing_number(child[l].ss, child[r].ss, middle + 1, right, s);
        return s;
    }
};

void solve() {
    int n, q; cin >> n >> q;
    string s;
    vi arr;
    vi st(n + 1);
    vi sz(1);
    for(int i = 1; i <= n; i++) {
        st[i] = s.size();
        string t; cin >> t;
        sz.pb(t.size());
        for(auto& ch : t) {
            s += ch;
            arr.pb(i);
        }
        s += '#';
        arr.pb(0);
    }
    const int N = s.size();
    suffix_array S(s);
    auto& pos = S.pos;
    auto& lcp = S.lcp;
    vi b(N);
    for(int i = 0; i < N; i++) {
        b[pos[i]] = arr[i];
    }
    wavelet_psgt tree(b);
    auto query = [&](int l, int r, int k) -> int {
        int x = pos[st[k]];
        int target = sz[k];
        int left = 0, right = x - 1, L = -1, R = x;
        while(left <= right) {
            int middle = midPoint;
            if(S.rmq.query(middle, x - 1) >= target) L = middle, right = middle - 1;
            else left = middle + 1;
        }
        if(L == -1) {
            if(S.lcp[x] < target) {
                return int(l <= k && k <= r);
            }
            L = x;
        }
        left = x, right = N - 1; 
        while(left <= right) {
            int middle = midPoint;
            if(S.rmq.query(x, middle) >= target) R = middle + 1, left = middle + 1;
            else right = middle - 1;
        }
        return tree.queries_range(L, R, l, r).cnt;
    };
    while(q--) {
        int l, r, k; cin >> l >> r >> k;
        cout << query(l, r, k) << '\n';
    }
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
