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
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;
template <int MOD>
struct mod_int {
    int value;
    
    mod_int(ll v = 0) { value = int(v % MOD); if (value < 0) value += MOD; }
    
    mod_int& operator+=(const mod_int &other) { value += other.value; if (value >= MOD) value -= MOD; return *this; }
    mod_int& operator-=(const mod_int &other) { value -= other.value; if (value < 0) value += MOD; return *this; }
    mod_int& operator*=(const mod_int &other) { value = int((ll)value * other.value % MOD); return *this; }
    mod_int pow(ll p) const { mod_int ans(1), a(*this); while (p) { if (p & 1) ans *= a; a *= a; p /= 2; } return ans; }
    
    mod_int inv() const { return pow(MOD - 2); }
    mod_int& operator/=(const mod_int &other) { return *this *= other.inv(); }
    
    friend mod_int operator+(mod_int a, const mod_int &b) { a += b; return a; }
    friend mod_int operator-(mod_int a, const mod_int &b) { a -= b; return a; }
    friend mod_int operator*(mod_int a, const mod_int &b) { a *= b; return a; }
    friend mod_int operator/(mod_int a, const mod_int &b) { a /= b; return a; }
    
    bool operator==(const mod_int &other) const { return value == other.value; }
    bool operator!=(const mod_int &other) const { return value != other.value; }
    bool operator<(const mod_int &other) const { return value < other.value; }
    bool operator>(const mod_int &other) const { return value > other.value; }
    bool operator<=(const mod_int &other) const { return value <= other.value; }
    bool operator>=(const mod_int &other) const { return value >= other.value; }
    
    mod_int operator&(const mod_int &other) const { return mod_int((ll)value & other.value); }
    mod_int& operator&=(const mod_int &other) { value &= other.value; return *this; }
    mod_int operator|(const mod_int &other) const { return mod_int((ll)value | other.value); }
    mod_int& operator|=(const mod_int &other) { value |= other.value; return *this; }
    mod_int operator^(const mod_int &other) const { return mod_int((ll)value ^ other.value); }
    mod_int& operator^=(const mod_int &other) { value ^= other.value; return *this; }
    mod_int operator<<(int shift) const { return mod_int(((ll)value << shift) % MOD); }
    mod_int& operator<<=(int shift) { value = int(((ll)value << shift) % MOD); return *this; }
    mod_int operator>>(int shift) const { return mod_int(value >> shift); }
    mod_int& operator>>=(int shift) { value >>= shift; return *this; }

    mod_int& operator++() { ++value; if (value >= MOD) value = 0; return *this; }
    mod_int operator++(int) { mod_int temp = *this; ++(*this); return temp; }
    mod_int& operator--() { if (value == 0) value = MOD - 1; else --value; return *this; }
    mod_int operator--(int) { mod_int temp = *this; --(*this); return temp; }

    explicit operator ll() const { return (ll)value; }
    explicit operator int() const { return value; }
    explicit operator db() const { return (db)value; }

    friend mod_int operator-(const mod_int &a) { return mod_int(0) - a; }
    friend ostream& operator<<(ostream &os, const mod_int &a) { os << a.value; return os; }
    friend istream& operator>>(istream &is, mod_int &a) { ll v; is >> v; a = mod_int(v); return is; }
};

const static int MOD = 1e9 + 7;
using mint = mod_int<MOD>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;
using pmm = pair<mint, mint>;
using vpmm = vt<pmm>;

vpll factorize_prime(ll n) {
    using u64  = uint64_t;
    using u128 = unsigned __int128;
    vll pf;

    auto mul_mod = [](u64 a, u64 b, u64 m) -> u64 {
        return (u64)((u128)a * b % m);
    };
    auto pow_mod = [&](u64 a, u64 e, u64 m) -> u64{
        u64 r = 1;
        while(e) { if (e & 1) r = mul_mod(r, a, m); a = mul_mod(a, a, m); e >>= 1; }
        return r;
    };
    auto isPrime = [&](u64 x)-> bool {
        if (x < 2) return false;
        for(u64 p:{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
            if(x % p == 0) return x == p;
        u64 d = x - 1, s = 0;
        while((d & 1) == 0) { d >>= 1; ++s; }
        for(u64 a:{2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL}) {
            u64 y = pow_mod(a, d, x);
            if(y == 1 || y == x - 1) continue;
            bool comp = true;
            for(u64 r = 1; r < s; ++r) {
                y = mul_mod(y, y, x);
                if(y == x - 1) { comp = false; break; }
            }
            if(comp) return false;
        }
        return true;
    };
    auto rho = [&](u64 n) -> u64{                
        if((n & 1) == 0) return 2;
        mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count()); 
        uniform_int_distribution<u64> dist(2, n - 2);
        while(true) {
            u64 y = dist(rng), c = dist(rng), m = 128, g = 1, r = 1, q = 1, ys, x;
            auto f = [&](u64 v){ return (mul_mod(v, v, n) + c) % n; };
            while(g == 1) {
                x = y;  for(u64 i=0; i < r; ++i) y = f(y);
                u64 k = 0;
                while(k < r && g == 1) {
                    ys = y;
                    u64 lim = min(m, r - k);
                    for(u64 i = 0; i < lim; ++i){ y = f(y); q = mul_mod(q, (x > y ? x - y : y - x), n); }
                    g = gcd(q, n);  k += m;
                }
                r <<= 1;
            }
            if(g == n) {
                do { ys = f(ys); g = gcd((x > ys ? x - ys : ys - x), n); } while (g == 1);
            }
            if(g != n) return g;
        }
    };

    auto fact = [&](auto& fact, u64 v) -> void {
        static const int small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43};
        for(int p : small){ if((u64)p * (u64)p > v) break;
            while(v % p == 0){ pf.pb(p); v /= p; }
        }
        if(v == 1) return;
        if(isPrime(v)){ pf.pb(v); return; }
        u64 d = rho(v);
        fact(fact, d); fact(fact, v / d);

    };

    if(n <= 0) return {};          
    fact(fact, (u64)n);
    srt(pf);
    vpll uniq;
    for(size_t i = 0; i < pf.size();) {
        size_t j = i; while(j < pf.size() && pf[j] == pf[i]) ++j;
        uniq.pb({pf[i], int(j - i)});
        i = j;
    }
    return uniq;
}
vll factorize_div(ll n) {
    using u64  = uint64_t;
    using u128 = unsigned __int128;
    vll pf;

    auto mul_mod = [](u64 a, u64 b, u64 m) -> u64 {
        return (u64)((u128)a * b % m);
    };
    auto pow_mod = [&](u64 a, u64 e, u64 m) -> u64{
        u64 r = 1;
        while(e) { if (e & 1) r = mul_mod(r, a, m); a = mul_mod(a, a, m); e >>= 1; }
        return r;
    };
    auto isPrime = [&](u64 x)-> bool {
        if (x < 2) return false;
        for(u64 p:{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
            if(x % p == 0) return x == p;
        u64 d = x - 1, s = 0;
        while((d & 1) == 0) { d >>= 1; ++s; }
        for(u64 a:{2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL}) {
            u64 y = pow_mod(a, d, x);
            if(y == 1 || y == x - 1) continue;
            bool comp = true;
            for(u64 r = 1; r < s; ++r) {
                y = mul_mod(y, y, x);
                if(y == x - 1) { comp = false; break; }
            }
            if(comp) return false;
        }
        return true;
    };
    auto rho = [&](u64 n) -> u64{                
        if((n & 1) == 0) return 2;
        mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count()); 
        uniform_int_distribution<u64> dist(2, n - 2);
        while(true) {
            u64 y = dist(rng), c = dist(rng), m = 128, g = 1, r = 1, q = 1, ys, x;
            auto f = [&](u64 v){ return (mul_mod(v, v, n) + c) % n; };
            while(g == 1) {
                x = y;  for(u64 i=0; i < r; ++i) y = f(y);
                u64 k = 0;
                while(k < r && g == 1) {
                    ys = y;
                    u64 lim = min(m, r - k);
                    for(u64 i = 0; i < lim; ++i){ y = f(y); q = mul_mod(q, (x > y ? x - y : y - x), n); }
                    g = gcd(q, n);  k += m;
                }
                r <<= 1;
            }
            if(g == n) {
                do { ys = f(ys); g = gcd((x > ys ? x - ys : ys - x), n); } while (g == 1);
            }
            if(g != n) return g;
        }
    };

    auto fact = [&](auto& fact, u64 v) -> void {
        static const int small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43};
        for(int p : small){ if((u64)p * (u64)p > v) break;
            while(v % p == 0){ pf.pb(p); v /= p; }
        }
        if(v == 1) return;
        if(isPrime(v)){ pf.pb(v); return; }
        u64 d = rho(v);
        fact(fact, d); fact(fact, v / d);

    };

    if(n <= 0) return {};          
    fact(fact, (u64)n);
    srt(pf);
    vpll uniq;
    for(size_t i = 0; i < pf.size();) {
        size_t j = i; while(j < pf.size() && pf[j] == pf[i]) ++j;
        uniq.pb({pf[i], int(j - i)});
        i = j;
    }
    vll divs = {1};
    for(auto [p, e] : uniq) {
        size_t sz = divs.size();
        ll pk = 1;
        for(int k = 1; k <= e;++k){
            pk *= p;
            for(size_t i = 0; i < sz; ++i) divs.pb(divs[i] * pk);
        }
    }
    srt(divs);
    return divs;
}

mint p[100];

mint count_ways(ll x, ll n) {
    if(n == 0) return x == 1;
    if(x == 1)  return 1;
    auto count_exponent = [](ll N, ll e) -> mint {
        if(e == 0) return  1;
        mint res = 1;
        for(int t = 0; t < e; t++) {
            res *= N + t;
        }
        return res * p[e];
    };
    mint ans = 1;
    for(auto& [p, e] : factorize_prime(x)) {
        ans *= count_exponent(n, e);
    }
    return ans;
}

void solve() {
    ll n, A, B; cin >> n >> A >> B;
    mint ways = 0;
    for(auto& x : factorize_div(B)) {
        if(x > A) break;
        ways += count_ways(x, n) * count_ways(B / x, n);
    }
    cout << ways << '\n';
}

signed main() {
    IOS;
    int t = 1;
    cin >> t;
    p[0] = 1;
    for(int i = 1; i < 100; i++) {
        p[i] = p[i - 1] * mint(i).inv();
    }
    for(int i = 1; i <= t; i++) {   
        cout << "Case #" << i << ": ";  
        solve();
    }
    return 0;
}
