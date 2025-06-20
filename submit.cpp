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
using mint = mod_int<998244353>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;
using pmm = pair<mint, mint>;
using vpmm = vt<pmm>;

struct BigCalculator {
    string num1;
    string num2;
    BigCalculator(const string n1, const string n2) : num1(n1), num2(n2) {}
    string add() { return bigAdd(num1, num2); }
    string mod() { return bigMod(num1, num2); }
    string minus() { return bigSub(num1, num2); }
    string multiply() { return bigMul(num1, num2); }
    string OR() { return bigBitOr(num1, num2); }
    string AND() { return bigBitAnd(num1, num2); }
    string XOR() { return bigBitXor(num1, num2); }
    string divide() { return bigDiv(num1, num2); }
    string operator|(const BigCalculator &other) const { return binToDec(bigBitOr(num1, other.num1)); }
    string operator&(const BigCalculator &other) const { return binToDec(bigBitAnd(num1, other.num1)); }
    string operator^(const BigCalculator &other) const { return binToDec(bigBitXor(num1, other.num1)); }
private:

    static string bigAdd(const string &a, const string &b) {
        int i = a.size() - 1, j = b.size() - 1, carry = 0;
        string result;
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) { sum += a[i] - '0'; i--; }
            if (j >= 0) { sum += b[j] - '0'; j--; }
            carry = sum / 10;
            result.pb('0' + sum % 10);
        }
        rev(result);
        return result;
    }
    static int compare(const string &a, const string &b) {
        if (a.size() != b.size())
            return (a.size() < b.size() ? -1 : 1);
        return a.compare(b);
    }
    static string bigSub(const string &a, const string &b) {
        if (compare(a, b) == 0) return "0";
        bool negative = false;
        string A = a, B = b;
        if (compare(a, b) < 0) { negative = true; swap(A, B); }
        int i = A.size() - 1, j = B.size() - 1, carry = 0;
        string result;
        while (i >= 0) {
            int diff = (A[i] - '0') - carry;
            if (j >= 0) { diff -= (B[j] - '0'); j--; }
            if (diff < 0) { diff += 10; carry = 1; }
            else { carry = 0; }
            result.pb('0' + diff);
            i--;
        }
        while (result.size() > 1 && result.back() == '0') result.pop_back();
        rev(result);
        if (negative) result.insert(result.begin(), '-');
        return result;
    }
    typedef complex<db> cd;
    static void fft(vector<cd> &a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            db ang = 2 * M_PI / len * (invert ? -1 : 1);
            cd wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                cd w(1);
                for (int j = 0; j < len / 2; j++) {
                    cd u = a[i + j];
                    cd v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            for (int i = 0; i < n; i++) a[i] /= n;
        }
    }
    static vector<int> multiplyFFT(const vector<int> &a, const vector<int> &b) {
        vt<cd> fa(all(a)), fb(all(b));
        int n = 1;
        while (n < int(a.size() + b.size())) n <<= 1;
        fa.resize(n); fb.resize(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++)
            fa[i] *= fb[i];
        fft(fa, true);
        vi result(n);
        int carry = 0;
        for (int i = 0; i < n; i++) {
            ll t = static_cast<ll>(round(fa[i].real())) + carry;
            result[i] = t % 10;
            carry = t / 10;
        }
        while (carry) {
            result.pb(carry % 10);
            carry /= 10;
        }
        while (result.size() > 1 && result.back() == 0)
            result.pop_back();
        return result;
    }
    static string bigMul(const string &a, const string &b) {
        if (a == "0" || b == "0")
            return "0";
        vi A(a.size()), B(b.size());
        for (size_t i = 0; i < a.size(); i++) A[a.size() - 1 - i] = a[i] - '0';
        for (size_t i = 0; i < b.size(); i++) B[b.size() - 1 - i] = b[i] - '0';
        auto resultVec = multiplyFFT(A, B);
        string result;
        for (int i = resultVec.size() - 1; i >= 0; i--) result.pb(resultVec[i] + '0');
        size_t pos = result.find_first_not_of('0');
        return (pos != string::npos) ? result.substr(pos) : "0";
    }
    static string decToBin(const string &dec) {
        if (dec == "0") return "0";
        string number = dec, bin;
        while (number != "0") {
            int rem;
            number = divideByTwo(number, rem);
            bin.pb('0' + rem);
        }
        rev(bin);
        return bin;
    }
    static string binToDec(const string &bin) {
        string result = "0";
        for (char bit : bin) {
            result = bigMul(result, "2");
            if (bit == '1') result = bigAdd(result, "1");
        }
        return result;
    }
    static string divideByTwo(const string &num, int &rem) {
        string quotient;
        int carry = 0;
        for (char c : num) {
            int current = carry * 10 + (c - '0');
            int digit = current / 2;
            carry = current % 2;
            if (!quotient.empty() || digit != 0) quotient.pb('0' + digit);
        }
        if (quotient.empty()) quotient = "0";
        rem = carry;
        return quotient;
    }
    static string bigBitOr(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.pb((binA[i] == '1' || binB[i] == '1') ? '1' : '0');
        return binRes;
    }
    static string bigBitAnd(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.pb((binA[i] == '1' && binB[i] == '1') ? '1' : '0');
        return binRes;
    }
    static string bigBitXor(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.pb((binA[i] != binB[i]) ? '1' : '0');
        return binRes;
    }

    static string bigDiv(const string &a, const string &b) {
        if (b.size() == 1) {
            int d = b[0] - '0';
            if (2 <= d && d <= 10) {
                string quotient;
                quotient.reserve(a.size());
                int rem = 0;
                for (char c : a) {
                    int v     = rem * 10 + (c - '0');
                    int digit = v / d;
                    rem       = v % d;
                    if (!quotient.empty() || digit > 0)
                        quotient.pb(char('0' + digit));
                }
                return quotient.empty() ? "0" : quotient;
            }
        }

        if (compare(a, b) < 0) return "0";
        string quotient, remainder = "0";
        for (char c : a) {
            remainder = bigAdd(bigMul(remainder, "10"), string(1, c));
            int lo = 0, hi = 9, q = 0;
            while (lo <= hi) {
                int mid  = (lo + hi) / 2;
                string p = bigMulSingle(b, mid);
                if (compare(p, remainder) <= 0) {
                    q  = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            quotient.pb(char('0' + q));
            remainder = bigSub(remainder, bigMulSingle(b, q));
        }
        int pos = 0;
        while (pos + 1 < (int)quotient.size() && quotient[pos] == '0') pos++;
        return quotient.substr(pos);
    }

    static string bigMod(const string &a, const string &b) {
        if (b.size() == 1) {
            int d = b[0] - '0';
            if (2 <= d && d <= 10) {
                int rem = 0;
                for (char c : a) {
                    rem = (rem * 10 + (c - '0')) % d;
                }
                return to_string(rem);
            }
        }

        if (compare(a, b) < 0) return a;
        string remainder = "0";
        for (char c : a) {
            remainder = bigAdd(bigMul(remainder, "10"), string(1, c));
            int lo = 0, hi = 9, q = 0;
            while (lo <= hi) {
                int mid  = (lo + hi) / 2;
                string p = bigMulSingle(b, mid);
                if (compare(p, remainder) <= 0) {
                    q  = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            remainder = bigSub(remainder, bigMulSingle(b, q));
        }
        return remainder;
    }

    static string bigMulSingle(const string &a, int d) {
        int carry = 0;
        string result;
        for(int i = a.size() - 1; i >= 0; i--){
            int prod = (a[i] - '0') * d + carry;
            carry = prod / 10;
            result.pb('0' + prod % 10);
        }
        while(carry){
            result.pb('0' + carry % 10);
            carry /= 10;
        }
        rev(result);
        int pos = 0;
        while(pos < result.size() - 1 && result[pos]=='0') pos++;
        return result.substr(pos);
    }
};

template<typename T>
pair<T,T> fib_pair(string n) { // return {f(n), f(n + 1)};
    if(n == "0") return {0, 1};
    auto [fk, fk1] = fib_pair<T>(BigCalculator(n, "2").divide());
    T c = fk * ((fk1 << 1) - fk);
    T d = fk * fk + fk1 * fk1;
    if((n.back() - '0') % 2 == 1) return {d, c + d};
    return {c, d};
}

void solve() {
    string n; cin >> n;
    cout << fib_pair<mint>(n).ff << '\n';;
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
