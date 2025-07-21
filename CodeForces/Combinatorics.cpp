vi primes, spf(MX), phi(MX);
vll gcd_sum(MX), lcm_sum(MX); // sum of gcd(i, k) for i from 1 to k
bitset<MX> primeBits;
vi mu(MX);

void generatePrime() {  
	primeBits.set(2);   
    for(int i = 3; i < MX; i += 2) primeBits.set(i);
    for(int i = 2; i * i < MX; i += (i == 2 ? 1 : 2)) {    
        if(primeBits[i]) {  
            for(int j = i; j * i < MX; j += 2) {    primeBits.reset(i * j); }
        }
    }
    for(int i = 0; i < MX; i++ ) {  if(primeBits[i]) {  primes.pb(i); } }   

//	iota(all(mu), 0); // for placeholder value
    // mu[1] = 1; // for count of occurences
    iota(all(phi), 0);
    for(int i = 1; i < MX; i++) {   
        if(!primeBits[i]) continue;
        for(int j = i; j < MX; j += i) {   
            // if(j >= i * 2) mu[j] -= mu[i];
            phi[j] -= phi[j] / i;
        }
    }
    mu[1] = 1;
    for(int i = 2; i < MX; i++) {
        if(spf[i] == 0) {
            for(int j = i; j < MX; j += i) {    if(spf[j] == 0) spf[j] = i; }
        }
        int p = spf[i];
        int m = i / p;
        mu[i] = m % p == 0 ? 0 : -mu[m];
    }
	for(int d = 1; d < MX; d++) {
        for(int j = d; j < MX; j += d) {
            gcd_sum[j] += phi[j / d] * (ll)d;
        }
    }

    for(int d = 1; d < MX; ++d) {
        ll term = (ll)d * phi[d];
        for(int n = d; n < MX; n += d) {
            lcm_sum[n] += term;
        }
    }
    for(int n = 1; n < MX; ++n) {
        lcm_sum[n] = (lcm_sum[n] + 1) * (ll)n / 2;
    }
} static const bool _generate_prime_init = []() { generatePrime(); return true; }();

// phi[divisor(n)] == n
//void insert(int x) {
//    for(auto& d : factorize(x)) {
//        ans += comb.choose(cnt[d]++, K - 1) * phi[d];
//    }    
//}

bool isPrime(uint64_t n) {
    if(n < 2) return false;
    for(uint64_t p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL})
        if(n % p == 0) return n == p;
    uint64_t d = n - 1, s = 0;
    while((d & 1) == 0) { d >>= 1; s++; }
    auto modpow = [&](uint64_t a, uint64_t e) {
        __uint128_t res = 1, base = a % n;
        while(e) {
            if(e & 1) res = (res * base) % n;
            base = (base * base) % n;
            e >>= 1;
        }
        return (uint64_t)res;
    };
    auto miller_pass = [&](uint64_t a) {
        uint64_t x = modpow(a, d);
        if(x == 1 || x == n-1) return true;
        for(uint64_t r = 1; r < s; r++) {
            x = (__uint128_t)x * x % n;
            if(x == n - 1) return true;
        }
        return false;
    };
    for(uint64_t a : {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL}) {
        if(a % n == 0) break;
        if(!miller_pass(a)) return false;
    }
    return true;
}

ll count_coprime(ll up, ll x) { // count number from [1 to up] where gcd(num, x) == 1
    auto d = factor_prime(x);
    int N = d.size();
    ll ans = 0;
    for (int mask = 0; mask < (1 << N); mask++) {
        int prod = up, sign = 1;
        for (int i = 0; i < N; i++) {
            if (have_bit(mask, i)) {
                prod /= d[i];
                sign *= -1;
            }
        }
        ans += prod * sign;
    }
    return ans;
}

vi factorize(int n, bool factor_prime = false) {
    vi divs;
    if(!factor_prime) {
        divs.pb(1);
    }
    while(n > 1) {
        int x = spf[n];
        int cnt = 0;
        while(n % x == 0) {
            n /= x;
            cnt++;
        }
        if(factor_prime) {
            divs.pb(x);
        } else {
            int d = 1;
            int N = divs.size();
            while(cnt--) {
                d *= x;
                for(int i = 0; i < N; i++) divs.pb(divs[i] * d);
            }
        }
    }
    return divs;
}

vll factorize(ll n) {
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

ll phi(ll n) {
    ll res = n;
    for(auto& [x, cnt] : factorize(n)) { // return primes factorize vector instead
        res -= res / x;
    }
    return res;
}

vi spf_factor(int n) {
    vi divs;
    divs.pb(1);
    while(n > 1) {
        int x = spf[n];
        int cnt = 0;
        while(n % x == 0) {
            n /= x;
            cnt++;
        }
        int d = 1;
        int N = divs.size();
        while(cnt--) {
            d *= x;
            for(int i = 0; i < N; i++) divs.pb(divs[i] * d);
        }
    }
    return divs;
}

template<typename T>
T totient_chain(const vpii& factors) {
    // given a number x in the form of [prime, power]
    // apply [int res = 0; while(x > 1) x = phi(x), res++; return res]
    // https://toph.co/p/eulers-peculiar-dream
    int maxp = 2;
    for(auto& [p, e] : factors) {
        maxp = max(maxp, p);
    }

    vt<T> cnt(maxp + 1, T(0));
    for(auto& [p, e] : factors) {
        cnt[p] += e;
    }
    // phi(p^e) where p is primes is p^e - p^(e - 1) = p^(e - 1) * (p - 1)
    // there are p^(e - 1) number where gcd is not 1 because since p is primes
    // there are p, 2p, 3p, 4p, (p^e)/p = p^(e - 1) number where gcd is not 1 so just take it off

    bool saw2 = (cnt.size() > 2 && cnt[2] != T(0));
    for(int i = maxp; i >= 3; --i) {
        if(spf[i] == i && cnt[i] != T(0)) {
            T ci = cnt[i];
            int x = i - 1;
            while(x > 1) {
                int f = spf[x];
                int c = 0;
                while(x % f == 0) {
                    x /= f;
                    ++c;
                }
                cnt[f] += ci * T(c);
            }
        }
    }

    T ans = cnt[2] + T(saw2 ? 0 : 1);
    return ans;
}

ll sum_of_divisors(ll num) {
    ll total = 1;
    for(ll i = 2; i * i <= num; i++) {
        if(num % i == 0) {
            int e = 0;
            do {
                e++;
                num /= i;
            } while (num % i == 0);

            ll sum = 0, pow = 1;
            do {
                sum += pow;
                pow *= i;
            } while (e-- > 0);
            total *= sum;
        }
    }
    if(num > 1) {
        total *= (1 + num);
    }
    return total;
}

struct NCkMod {
    static ll modPow(ll a, ll b, ll m) {
        ll r = 1; a %= m;
        while(b){ if(b & 1) r = r * a % m; a = a * a % m; b >>= 1; }
        return r;
    }
    static ll egcd(ll a, ll b, ll &x, ll &y) {
        if(!b){ x = 1; y = 0; return a; }
        ll g = egcd(b, a % b, y, x); y -= (a / b) * x; return g;
    }
    static ll modInv(ll a, ll m) {
        ll x, y; ll g = egcd(a, m, x, y);
        if(g != 1) return -1; return (x % m + m) % m;
    }
    using vec = vt<pair<ll, int>>;
    static vec factorize(int m) {
        vec f;
        for (int d = 2; d * d <= m; d++) if(m % d == 0){ int cnt = 0; while(m % d == 0){ cnt++; m /= d; } f.pb({d, cnt}); }
        if(m > 1) f.pb({m, 1}); return f;
    }
    static pll facto(ll n, int p, int pw, const vll& fact) {
        if(n < pw) {
            ll r = 1, e = 0;
            for (int i = 1; i <= n; i++){
                int x = i;
                while(x % p == 0){ e++; x /= p; }
                r = (r * x) % pw;
            }
            return {r, e};
        }
        auto sub = facto(n / p, p, pw, fact);
        ll r = modPow(fact[pw - 1], n / pw, pw);
        r = (r * fact[n % pw]) % pw;
        r = (r * sub.first) % pw;
        ll e = n / p + sub.second;
        return {r, e};
    }
    static ll binomPP(ll n, ll k, int p, int q) {
        int pw = 1; for (int i = 0; i < q; i++) pw *= p;
        vll fact(pw); fact[0] = 1;
        for (int i = 1; i < pw; i++)
            fact[i] = (i % p == 0 ? fact[i - 1] : fact[i - 1] * i % pw);
        auto A = facto(n, p, pw, fact);
        auto B = facto(k, p, pw, fact);
        auto C = facto(n - k, p, pw, fact);
        ll e = A.second - B.second - C.second;
        ll inv = modInv((B.first * C.first) % pw, pw);
        ll res = (A.first * inv) % pw;
        return (res * modPow(p, e, pw)) % pw;
    }
    static ll nck(ll n, ll k, int mod) {
        if(k < 0 || k > n) return 0;
        auto fac = factorize(mod);
        ll M = 1, ans = 0;
        vpll rem;
        for(auto &f : fac) {
            int p = f.first, q = f.second, pw = 1; for (int i = 0; i < q; i++) pw *= p;
            ll r = binomPP(n, k, p, q);
            rem.pb({r, pw}); M *= pw;
        }
        for(auto &r : rem) {
            ll m_i = r.second, a_i = r.first, M_i = M / m_i;
            ll inv = modInv(M_i, m_i);
            ans = (ans + a_i * M_i % M * inv) % M;
        }
        return ans;
    }
    static ll fastPowCheck(i128 base, i128 exp, ll limit) {
        i128 res = 1;
        while(exp > 0) {
            if(exp & 1) {
                res *= base;
                if(res > limit) return limit + 1;
            }
            exp >>= 1;
            if(exp > 0) {
                base = base * base;
                if((i128)base > limit) base = limit + 1;
            }
        }
        return (ll)res;
    }
    static ll nck_no_mod(ll n, ll k, ll limit = INF) {
        if(k < 0 || k > n) return 0;
        if(k > n - k) k = n - k;
        i128 ans = 1;
        for (int p : primes) {
            if(p > n) break;
            ll exp = 0;
            for (ll pp = p; pp <= n; pp *= p)
                exp += n / pp;
            for (ll pp = p; pp <= k; pp *= p)
                exp -= k / pp;
            for (ll pp = p; pp <= n - k; pp *= p)
                exp -= (n - k) / pp;
            ll factor = fastPowCheck(p, exp, limit);
            ans *= factor;
            if(ans > limit) return limit + 1;
        }
        return (ll)ans;
    }
};

ll get_perm(const vi& a, ll limit = INF) {
    // calculate (n!) / (x! * y! * z!...) where x + y + ... = n and don't mod upto limit
    ll total = sum(a), perm = 1;
    const int N = a.size();
    for(int i = 0; i < N; i++) {
        if(a[i] == 0) continue;
        perm *= NCkMod::nck_no_mod(total, a[i], limit);
        if(perm >= limit) {
            break;
        }
        total -= a[i];
    }
    return perm;
}

vi computeCatalan(int limit, int mod) {
    vi catalan(limit + 1, 0);
    catalan[0] = 1;
    for (int n = 1; n <= limit; n++) {
        ll temp = 0;
        for (int i = 0; i < n; i++) {
            temp = (temp + (ll)catalan[i] * catalan[n - 1 - i]) % mod;
        }
        catalan[n] = temp;
    }
    return catalan;
}

ll kth_coprime(ll p, int k, ll x = 0) {  // find the kth coprime starting where gcd(y, p) == 1 with the base of x
//    process this first for mobius function
//    mu[0] = 0;
//    mu[1] = 1;
//    for(int i = 2; i < MX; i++) {
//        if(primeBits[i]) {
//            mu[i] = -1;
//        } else {
//            int p = first_divisor[i];
//            int m = i / p;
//            if(m % p == 0) {
//                mu[i] = 0;
//            } else {
//                mu[i] = -mu[m];
//            }
//        }
//    }
    vi divs;
    for(int i = 1; 1LL * i * i <= p; i++){
        if(p % i == 0) {
            divs.pb(i);
            if(i * i != p) divs.pb(p / i);
        }
    }
    auto cal = [&](ll y) -> ll {
        ll cnt = 0;
        for(int d : divs){
            cnt += mu[d] * (y / d);
        }
        return cnt;
    };
    ll base = cal(x);
    ll target = base + k;

    ll lo = x + 1, hi = x + k;
    while(cal(hi) < target){
        hi = x + (hi - x) * 2;  
    }
    ll ans = hi;
    while(lo <= hi){
        ll mid = (lo + hi) >> 1;
        if(cal(mid) >= target){
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return ans;
}

template<class T> 
class Combinatoric {    
    public: 
    int n;  
    vt<T> fact, inv;   
    Combinatoric(int n) {   
        this->n = n;    
        fact.rsz(n + 1), inv.rsz(n + 1);
        init();
    }
        
    void init() {   
        fact[0] = 1;
        for(int i = 1; i <= n; i++) {   
            fact[i] = fact[i - 1] * i;
        }
        inv[n] = fact[n].inv();
        for(int i = n - 1; i >= 0; i--) {   
            inv[i] = inv[i + 1] * (i + 1);
        }
    }
    
    T choose(int a, int b) {  
        if(a < b) return 0;
        assert(max(a, b) <= n);
        return fact[a] * inv[b] * inv[a - b];
    }
	
    T nCk(int n, int r) { // change to ll if needed
        if(n < r) return 0;
		T ans = 1;
        for(int i = 1 ; i <= r ; i++) {
            ans *= n - i + 1;
            ans /= i ;   
        }
        return ans ;
    }

	T nCk_increasing_sequence(int l, int r, int len) { // given a range of number from l to r, len k, 
                                                       // return the number of ways to choose those element in increasing order
//        if(len > r - l + 1) return 0;  // not enough numbers
//        return choose(r - l + 1, len); // for strictly increasing/decreasing
        return choose(r - l + len, len);
        // x _ _ _ y
        // # of way to choose the _ unknown value
        // len = pos[y] - pos[x] - 1
    }


//    ll nCk_mod_Lucas_Theorem(int n, int r, int mod) {
//        if(r > n) return 0 ;
//        ll res = 1;
//        while(n && r) {
//            res *= nCk(n % mod, r % mod) ;
//            res %= mod ;
//            n /= mod ;
//            r /= mod ; 
//        }
//        return res ;
//    }
//
//    int nCk_lucas(int n, int r, int mod) {
//        vi ans;
//        for(auto& x : DIV[mod]) {
//            ans.pb(nCk_mod_Lucas_Theorem(n, r, x));
//        }
//        ll res = 0;
//        for(int i = 0; i < int(DIV[mod].size()); i++) {
//            int p = DIV[mod][i];
//            ll m = mod / p;
//            ll inv = modExpo(m, p - 2, p);
//            res = (res + ans[i] * m % mod * inv) % mod;
//        }
//        return res;
//    }

    T catalan(int k) { // # of pair of balanced bracket of length n is catalan(n / 2)
        if(k == 0) return 1;
        return choose(2 * k, k) - choose(2 * k, k - 1);
    }

	T monotonic_array_count(int n, int m) {// len n, element from 1 to m increasing/decreasing
        return choose(n + m - 1, n);
    }

}; Combinatoric<mint> comb(MX);

struct SumKBinomial {
    // https://codeforces.com/contest/1549/problem/E
    // return sum (nck(1 * k, x) + nck(2 * k, x) + ... + (n * k, x)) for each x from 1 to n * k
    // bruteforce is given x
    // for(int i = 0; i <= n; i++) {
    //      ans += nck(i * k, x)
    // }
    int N, K;
    vt<T> A;
    SumKBinomial(int N_, int K_) : N(N_), K(K_), A(K_ * N_ + 1) {
        T invK = T(K).inv();
        A[0] = T(N);
        int M = K * N;
        for(int j = 1; j <= M; j++) {
            T num = comb.choose(K * (N + 1), j + 1) - comb.choose(K, j + 1);
            for(int i = 2; i <= K; i++) {
                int idx = j + 1 - i;
                if(idx < 0) break;
                num -= comb.choose(K, i) * A[idx];
            }
            A[j] = num * invK;
        }
    }

    T query(int x) const {
        return (x >= 0 && x < (int)A.size() ? A[x] : T(0));
    }
};

template<typename T>
T sum_of_powers(long long n, int k) { // find (1 ^ k + 2 ^ k + ... + n ^ k) sum
    // https://codeforces.com/contest/622/problem/F
    int M = k + 1;
    vt<T> y(M + 1);
    y[0] = T(0);
    for(int i = 1; i <= M; i++) {
        y[i] = y[i - 1] + T(i).pow(k);
    }
    if(n <= M) return y[n];
    vt<T> pref(M + 1), suf(M + 1);
    pref[0] = T(1);
    for(int i = 1; i <= M; i++) {
        pref[i] = pref[i - 1] * T(n - (i - 1));
    }
    suf[M] = T(1);
    for (int i = M-1; i >= 0; i--) {
        suf[i] = suf[i + 1] * T(n - (i + 1));
    }
    T ans = T(0);
    for(int i = 0; i <= M; i++) {
        T num = pref[i] * suf[i];
        T invden = comb.inv[i] * comb.inv[M - i];
        if((M - i) & 1) invden = -invden;
        ans += y[i] * num * invden;
    }
    return ans;
}

// pascal triangle
// dp[n][k] = dp[n - 1][k] + dp[n - 1][k - 1];
// for nck sweep line, we go from highest k to 0
//        for(int j = k; j < K; j++) { // because it's not normal sweepline for updating from [l, r - 1]
//            for(auto& [l, r] : Q[j]) {
//                int d = j - k;
//                dp[r] -= comb.choose(r - l - 1 + d, d);
//            }
//        }

//        dp[i] = comb.choose(r + c - 2, c - 1);
//        for(int j = 0; j < i; j++) {
//            if(a[j].ff <= r && a[j].ss <= c) {
//                int nr = r - a[j].ff, nc = c - a[j].ss;
//                dp[i] -= dp[j] * comb.choose(nr + nc, nr);
//            }
//        }

ll missing(const ll& target, const ll& x) { // minimum +1 operation to make x a superset of target, (target & x) = target
    ll diff = target & ~x;
    if(diff == 0) return 0;
    int msb = max_bit(diff);
    const ll N = 1LL << msb;
    ll carry_cost = N - (x & (N - 1)); // fill in the msb bit, now everything below msb become 0 in x
    ll restore_cost = ((diff | target) & (N - 1)); // restore the missing bit < msb
    return carry_cost + restore_cost;
}

ll XOR(ll n) {    
	if(n % 4 == 0) return n;
	if(n % 4 == 1) return 1;    
	if(n % 4 == 2) return n + 1;
	return 0;
};

ll AND(ll l, ll r) {
    int shift = 0;
    while (l < r) {
        l >>= 1;
        r >>= 1;
        shift++;
    }
    return l << shift;
}

ll OR(ll l, ll r) {
    int shift = 0;
    while (l < r) {
        l >>= 1;
        r >>= 1;
        shift++;
    }
    return (l << shift) | ((1 << shift) - 1);
}

vll countBit(ll n) {    
	int m = 62;
	vll cnt(m);
    auto f = [&](ll A, ll B) -> ll {
        return A + B;
    };
	while(n > 0) {  
		ll msb = log2(n);
		ll c = (1LL << msb); 
		cnt[msb] = f(cnt[msb], n - c + 1);
		n -= c;
		c >>= 1;
		for(int i = 0; i < msb; i++) {  
			cnt[i] += c; // careful with MOD
		}
	}
	return cnt;
};

ll get_prime_mask(ll n) {
    ll mask = 1;
    for(auto& x : primes) {
        if(x * x > n) break;
		if(n % x) continue;
        int p = 0;
        while(n % x == 0) {
            p ^= 1;
            n /= x;
        }
        if(p) mask *= x;
    }
    if(n > 1) mask *= n;
    return mask;
}

string get_base_k(string n, int k) {
    if(n == "0") return "0";
    string s;
    while(n != "0") {
        string S;
        int rem = 0;
        for (char ch : n) {
            int curr = rem * 10 + (ch - '0');
            int qDigit = curr / k;
            rem = curr % k;
            if(!S.empty() || qDigit != 0)
                S.pb(qDigit + '0');
        }
        s.pb(rem + '0');
        n = S.empty() ? "0" : S;
    }
    rev(s);
    return s;
}

string get_base_negk_to_string(ll n, ll k) {
    if(n == 0) return "0";
    string s;
    while(n) {
        ll r = n % (-k);
        n /= (-k);
        if(r < 0) {
            r += k;
            n++;
        }
        s.push_back('0' + r);
    }
    rev(s);
    return s;
}

ll get_mask(ll a, ll k) { // get bit_mask representation in base k
    ll res = 0;
    int cnt = 0;
    while(a > 0) {
        ll digit = a % k;
        if(digit > 1) return -1;
        if(digit == 1) res |= (1LL << cnt);
        a /= k;
        cnt++;
    }
    return res;
}

// Does the inverse of `submask_sums`; returns the input that produces the given output.
template<typename T_out, typename T_in>
void mobius_transform(int n, vt<T_in> &values) { // remember to set dp[mask] = -dp[mask] if(pct(mask) % 2 == 0) later
    assert(int(values.size()) == 1 << n);
 
    for (int i = 0; i < n; i++) {
        for (int base = 0; base < 1 << n; base += 1 << (i + 1)) {
            for (int mask = base; mask < base + (1 << i); mask++) {
                values[mask + (1 << i)] -= values[mask];
            }
        }
    }
}

pll find_any_solution(ll a, ll b, ll x) { // find [c, d] such that a * c + b * d = x
    ll X, Y;
    ll g = extended_gcd(a, b, X, Y);
    if(x % g != 0) return {-INF, -INF};
    ll factor = x / g;
    ll c0 = X * factor;
    ll d0 = Y * factor;
    return {c0, d0};
}

bool find_solution(ll a, ll b, ll mna, ll mxa, ll mnb, ll mxb, ll target, ll &x_out, ll &y_out) { // find solution within range
    ll x0, y0;
    ll g = extended_gcd(a, b, x0, y0);
    if(target % g) return false;
    x0 *= target / g;
    y0 *= target / g;
    ll shiftX = b / g;
    ll shiftY = a / g;
    ll t1 = ceil(mna - x0, shiftX);
    ll t2 = floor(mxa - x0, shiftX);
    if(t1 > t2) return false;
    ll t3 = ceil(y0 - mxb, shiftY);
    ll t4 = floor(y0 - mnb, shiftY);
    if(t3 > t4) return false;
    ll t_low  = max(t1, t3);
    ll t_high = min(t2, t4);
    if(t_low > t_high) return false;
    x_out = x0 + shiftX * t_low;
    y_out = y0 - shiftY * t_low;
    return true;
}

bool crt(ll r1, ll m1, ll r2, ll m2, ll&k, ll& m) { // find k < lcm(m1, m2) such that k % m1 == r1 && k % m2 == r2
    r1 = (r1 % m1 + m1) % m1;
    r2 = (r2 % m2 + m2) % m2;
    auto sol = find_any_solution(m1, -m2, r2 - r1);
    if(sol.ff == -INF && sol.ss == -INF) return false; // careful for collision
    ll t1 = sol.ff;
    m = lcm(m1, m2); // adjust as needed
    k = (r1 + (m1 % m * t1 % m) % m) % m;
    if(k < 0) k += m;
    return true;
}

void shift_solution(ll &x, ll &y, ll a, ll b, ll cnt) {
  x += cnt * b;
  y -= cnt * a;
}
// returns the number of solutions where x is in the range[minx, maxx] and y is in the range[miny, maxy]
ll find_all_solutions(ll a, ll b, ll c, ll minx, ll maxx, ll miny,ll maxy) {
	// https://github.com/ShahjalalShohag/code-library/blob/main/Number%20Theory/Linear%20Diophantine%20Equation%20with%20Two%20Variables.cpp
  ll x, y, g;
  if (find_any_solution(a, b, c, x, y, g) == 0) return 0;
  if (a == 0 and b == 0) {
    assert(c == 0);
    return 1LL * (maxx - minx + 1) * (maxy - miny + 1);
  }
  if (a == 0) {
    return (maxx - minx + 1) * (miny <= c / b and c / b <= maxy);
  }  
  if (b == 0) {
    return (maxy - miny + 1) * (minx <= c / a and c / a <= maxx);
  }
  a /= g, b /= g;
  ll sign_a = a > 0 ? +1 : -1;
  ll sign_b = b > 0 ? +1 : -1;
  shift_solution(x, y, a, b, (minx - x) / b);
  if (x < minx) shift_solution(x, y, a, b, sign_b);
  if (x > maxx) return 0;
  ll lx1 = x;
  shift_solution(x, y, a, b, (maxx - x) / b);
  if (x > maxx) shift_solution (x, y, a, b, -sign_b);
  ll rx1 = x;
  shift_solution(x, y, a, b, -(miny - y) / a);
  if (y < miny) shift_solution (x, y, a, b, -sign_a);
  if (y > maxy) return 0;
  ll lx2 = x;
  shift_solution(x, y, a, b, -(maxy - y) / a);
  if (y > maxy) shift_solution(x, y, a, b, sign_a);
  ll rx2 = x;
  if (lx2 > rx2) swap (lx2, rx2);
  ll lx = max(lx1, lx2);
  ll rx = min(rx1, rx2);
  if (lx > rx) return 0;
  return (rx - lx) / abs(b) + 1;
}

ll solveCRT(const vpii& congruences) { // return a value x such that x % a[i].ff == a[i].ss for all i
    ll r0 = 0, m0 = 1;
    for (auto &[p, r] : congruences) {
        ll s, t;
        ll g = extended_gcd(m0, p, s, t);
        if((r - r0) % g != 0) return -1;
        ll mod = p / g;
        ll k = ((r - r0) / g % mod * (s % mod)) % mod;
        if(k < 0) k += mod;
        r0 = r0 + m0 * k;
        m0 = m0 / g * p;
        r0 %= m0;
        if(r0 < 0) r0 += m0;
    }
    return r0;
}

pll solve_formula(ll a, ll b, ll c, ll d) {
    if((i128)a * d >= (i128)b * c) return {-INF, -INF};
	ll n = a / b;
	a -= n * b, c -= n * d;
	if(c > d) return {n + 1, 1};
	auto [p, q] = solve_formula(d, c, b, a);
	return {p * n + q, p};
}

vi get_pair_gcd(vi& a) {
    int m = MAX(a);
    vi f(m + 1), g(m + 1);
    for(auto& x : a) f[x]++;
    for(int i = 1; i <= m; i++) {
        for(int j = i; j <= m; j += i) {
            g[i] += f[j];
        }
    }
    for(int i = m; i >= 1; i--) {
        for(int j = i * 2; j <= m; j += i) {
            g[i] -= g[j];
        }
    }
    return g;
}

int find_y(int x, int p, int c) { // find y such that (x * y) % p == c
    auto egcd = [&](auto self, int a, int b) -> tuple<int, int, int> {
        if (!b) return {a, 1, 0};
        auto [g, s, t] = self(self, b, a % b);
        return make_tuple(g, t, s - (a / b) * t);
    };

    auto modInv = [&](int a, int m) -> int {
        auto [d, inv, _] = egcd(egcd, a, m);
        if (d != 1) return -1;
        inv %= m;
        return inv < 0 ? inv + m : inv;
    };

    auto [d, s, t] = egcd(egcd, x, p);
    if (c % d != 0) return -1;
    int x1 = x / d, p1 = p / d, c1 = c / d;
    int inv = modInv(x1, p1);
    if (inv == -1) return -1;
    int y0 = (ll)c1 * inv % p1;
    return y0 < 0 ? y0 + p1 : y0;
}

template<typename T>
struct sos_dp {
    ll B, N;
    vt<T> subset, superset, f, a;
    sos_dp(const vt<T>& a) : a(a) {
        B = 0;
        ll m = MAX(a);
        while((1LL << B) <= m) B++;
        N = 1LL << B;
        f.rsz(N);
    }
    sos_dp(const vt<T>& a, ll B) : a(a), B(B), N(1LL << B), f(1LL << B) {}

    ll low;
    sos_dp(int _B, int low) : low(low), B(_B), N(1LL << _B) { } // meet in the middle, https://www.codechef.com/problems/MONSTER?tab=statement

    void load() {
        f = a;
    }
    // how many element have a[i] as a super set
    void subsetSOS() {
        load();
        assert(subset.empty());
        for(ll bit = 0; bit < B; bit++){
            for(ll mask = 0; mask < N; mask++){
                if(have_bit(mask, bit)){
                    f[mask] += f[mask ^ (1 << bit)];
                }
            }
        }
        subset = f;
    }

    // how many element is a[i] a subset of
    // how many element AND together have the freq of it is
    // freq[nxt] - freq[mask] where have_bit(mask, bit) and nxt = (mask ^ (1LL << bit))
    // see max_prefix_sum_and function for reference
    void supersetSOS() {
        load();
        assert(superset.empty());
        for(ll bit = 0; bit < B; bit++){
            for(ll mask = 0; mask < N; mask++){
                if(!have_bit(mask, bit)){
                    f[mask] += f[mask | (1 << bit)];
                }
            }
        }
        superset = f;
    }

    template<typename F>
    F subset_equal_to(ll target) { // how many OR subset equal to target
        // https://www.hackerrank.com/contests/w16/challenges/vim-war/problem
        subsetSOS();
        F res = 0;
        for(ll mask = target; ; mask = (mask - 1) & target) {
            if(pct(mask ^ target) & 1) res -= F(2).pow(f[mask]) - 1; // contribution of mask to target is (mask ^ target)
            else res += F(2).pow(f[mask]) - 1;
            if(mask == 0) break;
        }
        return res;
    }

    vll A;
    void update_subset(ll mask, ll delta = 1) { // tc : 1 << (K - low)
        if(A.empty()) A.rsz(N);
        ll lo = ((1LL << low) - 1) & mask;
        mask = (mask >> low) << low;
        for(ll sub = mask; ; sub = (sub - 1) & mask) {
            ll now = sub | lo;
            if(now < A.size()) A[now] += delta;
            if(sub == 0) break;
        }
    }

    ll query_subset(ll mask) { 
        if(A.empty()) return 0;
        const ll LOW = (1LL << low) - 1;
        ll res = 0;
        ll hi = (mask >> low) << low;
        ll orig_low = mask & LOW;
        mask = ~mask & LOW;
        for(ll sub = mask;; sub = (sub - 1) & mask) {
            ll now = hi | orig_low | sub;
            if(now < A.size()) res += A[now];
            if(sub == 0) break;
        }
        return res;
    }

    void update_superset(ll mask, ll delta = 1) {
        if(A.empty()) A.rsz(N);
        ll lo = ((1LL << low) - 1) & mask;
        const ll LOW = (((~mask & (N - 1)) >> low) << low);
        for(ll sub = LOW; ; sub = (sub - 1) & LOW) {
            ll now = sub | mask;
            if(now < A.size()) A[now] += delta;
            if(sub == 0) break;
        }
    }

    ll query_superset(ll mask) {
        if(A.empty()) return 0;
        const ll LOW = ((1LL << low) - 1) & mask;
        mask = (mask >> low) << low;
        ll res = 0;
        for(ll sub = LOW; ; sub = (sub - 1) & LOW) {
            ll now = sub | mask;
            if(now < A.size()) res += A[now];
            if(sub == 0) break;
        }
        return res;
    }

    ll max_prefix_sum_and() { // max sum of every prefix and over all permutation
        supersetSOS();
        vll dp(N, -INF);
        dp[N - 1] = (ll)count(all(a), N - 1) * (N - 1);
        for(ll mask = N - 1; mask >= 0; mask--) {
            if(dp[mask] == -INF) continue;
            for(ll b = 0; b < B; b++) {
                if(have_bit(mask, b)) {
                    ll nxt = mask ^ (1 << b);
                    ll cnt = f[nxt] - f[mask];
                    dp[nxt] = max(dp[nxt], dp[mask] + cnt * nxt);
                }
            }
        }
        return dp[0];
    }

    ll calc_submask(ll one, ll zero, ll question) { // calculate how many submask that has 0 and 1 bits like one and zero mask
                                                    // in addition to that, question_mask can be 0 and 1 as well, marking it 2 ^ (pct(question)) mask
        // https://oj.uz/problem/view/JOI18_snake_escaping
        // remember to reverse the string if needed
        if(subset.empty()) subsetSOS();
        if(superset.empty()) supersetSOS();
        const ll lim = B / 3;
        ll ans = 0;
        if(pct(question) <= lim) {
            for(ll sub = question; ; sub = (sub - 1) & question) {
                ans += a[sub | one];
                if(sub == 0) break;
            }
        } else if(pct(zero) <= lim) {
            for(ll sub = zero; ; sub = (sub - 1) & zero) {
                ll coeff = (pct(sub) & 1) ? -1 : 1;
                ans += coeff * superset[sub | one];
                if(sub == 0) break;
            }
        } else {
            for(ll sub = one; ; sub = (sub - 1) & one) {
                ll coeff = ((pct(sub) ^ pct(one)) & 1) ? -1 : 1;
                ans += coeff * subset[sub | question];
                if(sub == 0) break;
            }
        }
        return ans;
    }
};

vll submask_nck(int n, int k) {
    if(k < 0 || k > n) return {};
    if(k == 0) return {0};
    vll masks;
    ll m = (1LL << k) - 1;
    ll limit = 1LL << n;
    while(m < limit) {
        masks.pb(m);
        ll x = m & -m;
        ll y = m + x;
        m = ((m & ~y) / x >> 1) | y;
    }
    return masks;
}


//for(auto& p : primes) {
//	for(int c = m / p; c >= 1; c--) f[c] += f[c * p];
//}
//for (int d = m; d >= 1; d--) {
//    dp[d] = f[d] * d;
//    for (auto &p : primes) {
//        if ((ll)d * p > m) break;
//        dp[d] = max(dp[d], dp[d * p] + (f[d] - f[d * p]) * d);
//    }
//}
//same as 
//for(int i = 1; i <= m; i++) {
//	for(int j = i * 2; j <= m; j++) {
//		g[i] += g[j];
//	}
//}
//for(ll i = m; i >= 1; i--) {
//    dp[i] = f[i] * i;
//    for(int j = i * 2; j <= m; j += i) {
//        dp[i] = max(dp[i], dp[j] + (f[i] - f[j]) * i);
//    }
//}



// number of ways to make up the sum of s is 2 ^ (s - 1)

//    ll curr = 1;
//    while(curr <= j) {
//        ll add = j / curr;
//        ll last = j / add;
//        if(j + add > MX) break;
//        if(dp[j + add] == -1) {
//            dp[j + add] = dp[j] + 1;
//            q.push(j + add);
//        }
//        curr = last + 1;
//    }

//    int n; cin >> n; // compute distinct prefix_or of an array over all permutation
//    const int N = 2 * n;
//    vi mask(N), dp(N);
//    int zero = 0;
//    while(n--) {
//        int x; cin >> x;
//        zero |= x == 0;
//        mask[x] = x;
//    }
//    const int K = 22;
//    for(int bit = 0; bit < K; bit++) {
//        for(int x = 0;x < N; x++) {
//            if((x >> bit) & 1) mask[x] |= mask[x ^ (1 << bit)];
//        }
//    }
//    for(int x = 0; x < N; x++) {
//        for(int bit = 0; bit < K; bit++) {
//            if((x >> bit) & 1) {
//                int v = dp[x ^ (1 << bit)];
//                if((mask[x] >> bit) & 1) v++;
//                dp[x] = max(dp[x], v);
//            }
//        }
//    }
//    int res = MAX(dp) + zero;
//    cout << res << endl;