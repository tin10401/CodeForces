#include<bits/stdc++.h>
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
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 1e18;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
ll gcd(ll a, ll b) { while (b != 0) { ll temp = b; b = a % b; a = temp; } return a; }
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
ll floor(ll a, ll b) { if(b < 0) a = -a, b = -b; if (a >= 0) return a / b; return a / b - (a % b ? 1 : 0); }
ll ceil(ll a, ll b) { if (b < 0) a = -a, b = -b; if (a >= 0) return (a + b - 1) / b; return a / b; }
int pct(ll x) { return __builtin_popcountll(x); }
int have_bit(ll x, int b) { return (x >> b) & 1; }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const string dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) { ll m = (n + 1) / 2; return m * m; }
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^0 + p^1 + p^2 + p^3 + ... + p^k
template<typename T> T geometric_power_range(T base, ll startExp, ll endExp) { // return base^startExp + base^(startExp + 1) + ... + base^endExp
    if(startExp > endExp) return 0;
    T first = base.pow(startExp);
    ll len = endExp - startExp + 1;
    return first * (base.pow(len) - 1) / (base - 1);
}
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
//bool is_two_prime_sum(ll n) { return n >= 4 && (n % 2 == 0 || isPrime(n - 2)); }
//bool is_three_prime_sum(ll n) { return n >= 6 && (n % 2 || is_two_prime_sum(n - 2)); }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}

template<typename T, int BITS = 30>
struct xor_basis {
    // subsequence [l, r] having subsequence_xor of x is pow(2, (r - l + 1) - rank())
    T basis[BITS];
    int r = 0;
 
    bool insert(T x) {
        if (x == 0) return false;
        x = min_value(x);
        if (x == 0) return false;
        for (int i = 0; i < r; ++i)
            if ((basis[i] ^ x) < basis[i]) basis[i] ^= x;
        basis[r++] = x;
        int k = r - 1;
        while (k > 0 && max_bit(basis[k]) > max_bit(basis[k - 1])) {
            swap(basis[k], basis[k - 1]);
            --k;
        }
        return true;
    }
 
    bool contains(T x) const {
        return min_value(x) == 0;
    }
 
    T min_value(T x) const {
        if(r == BITS) return 0;
        for (int i = 0; i < r; ++i) x = min(x, x ^ basis[i]);
        return x;
    }
 
    T max_value(T x = 0) const {
        for (int i = 0; i < r; ++i) x = max(x, x ^ basis[i]);
        return x;
    }
 
    int rank() const { return r; }
 
    uint64_t size() const { return (r >= 64 ? 0ULL : 1ULL << r); }
 
    vt<T> get_compact_basis() const {
        vt<T> vec;
        for (int i = 0; i < r; ++i) vec.pb(basis[i]);
        return vec;
    }
 
    T get_kth_smallest(uint64_t k) const {
        if (r >= 64 || k >= (1ULL << r)) return T(0);
        T ans = 0;
        for (int i = 0; i < r; ++i)
            if ((k >> i) & 1ULL) ans ^= basis[i];
        return ans;
    }
 
    T get_kth_largest(uint64_t k) const {
        uint64_t total = size();
        if (total == 0 || k >= total) return T(0);
        return get_kth_smallest(total - 1 - k);
    }
 
    bool insert_base_on(T x, T c) {
        for (int i = 0; i < r; ++i) x = std::min(x, x ^ basis[i]);
        if (x == 0) return false;
        for (int b = BITS - 1; b >= 0; --b) {
            if (have_bit(c, b) || !have_bit(x, b)) continue;
            for (int i = 0; i < r; ++i)
                if (have_bit(basis[i], b)) basis[i] ^= x;
            basis[r++] = x;
            int k = r - 1;
            while (k > 0 && max_bit(basis[k]) > max_bit(basis[k - 1])) {
                std::swap(basis[k], basis[k - 1]);
                --k;
            }
            return true;
        }
        return false;
    }
 
    T max_value_base_on(T x) const {
        T res = 0;
        for (int i = 0; i < r; ++i)
            if (!have_bit(x, max_bit(basis[i]))) res ^= basis[i];
        return res;
    }

    template<typename F>
    F subsequence_xor_contain_x(ll x, int len) { // https://codeforces.com/contest/959/problem/F
        // there are rank() important value, (len - rank()) non-important value, which can be expressed as some subset
        // can skip or take, either way we can still express them as a valid xor == 0 then xor with x
        return contains(x) ? F(2).pow(len - rank()) : 0;
    }
 
    inline xor_basis operator+(const xor_basis& other) const {
        if (r == 0) return other;
        if (other.r == 0) return *this;
        const xor_basis* big   = (r >= other.r ? this  : &other);
        const xor_basis* small = (r >= other.r ? &other :  this);
        xor_basis res = *big;
        for (int i = 0; i < small->r; ++i) res.insert(small->basis[i]);
        return res;
    }
 
    xor_basis& operator^=(T k) {
        for (int i = r - 1; i >= 0; --i)
            if (basis[i] & 1) {
                basis[i] ^= k;
            }
        return *this;
    }
 
    bool operator==(const xor_basis &o) const {
        if (r != o.r) return false;
        for (int i = 0; i < r; ++i)
            if (basis[i] != o.basis[i]) return false;
        return true;
    }
};

void solve() {
    // from 0, you can reach any subset of xi by continuously xor-ing, creating 1 connecting component
    // each component will have size 1 << rank(), total is 1 << k
    // answer = (1 << k) / (1 << rank()) = 1 << (k - rank())
    int k, n; cin >> k >> n;
    xor_basis<int> xr;
    for(int i = 0; i < n; i++) {
        int x; cin >> x;
        xr.insert(x);
    }
    cout << (1LL << (k - xr.rank())) << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
