template <int MOD>
struct mod_int {
    int value;
    
    mod_int(long long v = 0) { value = int(v % MOD); if (value < 0) value += MOD; }
    
    mod_int& operator+=(const mod_int &other) { value += other.value; if (value >= MOD) value -= MOD; return *this; }
    mod_int& operator-=(const mod_int &other) { value -= other.value; if (value < 0) value += MOD; return *this; }
    mod_int& operator*=(const mod_int &other) { value = int((long long)value * other.value % MOD); return *this; }
    mod_int pow(long long p) const { mod_int ans(1), a(*this); while (p) { if (p & 1) ans *= a; a *= a; p /= 2; } return ans; }
    
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
    
    mod_int operator&(const mod_int &other) const { return mod_int((long long)value & other.value); }
    mod_int& operator&=(const mod_int &other) { value &= other.value; return *this; }
    mod_int operator|(const mod_int &other) const { return mod_int((long long)value | other.value); }
    mod_int& operator|=(const mod_int &other) { value |= other.value; return *this; }
    mod_int operator^(const mod_int &other) const { return mod_int((long long)value ^ other.value); }
    mod_int& operator^=(const mod_int &other) { value ^= other.value; return *this; }
    mod_int operator<<(int shift) const { return mod_int(((long long)value << shift) % MOD); }
    mod_int& operator<<=(int shift) { value = int(((long long)value << shift) % MOD); return *this; }
    mod_int operator>>(int shift) const { return mod_int(value >> shift); }
    mod_int& operator>>=(int shift) { value >>= shift; return *this; }

    mod_int& operator++() { ++value; if (value >= MOD) value = 0; return *this; }
    mod_int operator++(int) { mod_int temp = *this; ++(*this); return temp; }
    mod_int& operator--() { if (value == 0) value = MOD - 1; else --value; return *this; }
    mod_int operator--(int) { mod_int temp = *this; --(*this); return temp; }

    explicit operator ll() const { return value; }
    explicit operator int() const { return value; }
    explicit operator db() const { return value; }

    friend mod_int operator-(const mod_int &a) { return mod_int(0) - a; }
    friend std::ostream& operator<<(std::ostream &os, const mod_int &a) { os << a.value; return os; }
    friend std::istream& operator>>(std::istream &is, mod_int &a) { long long v; is >> v; a = mod_int(v); return is; }
};

const static int MOD = 1e9 + 7;
using mint = mod_int<998244353>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;
using pmm = pair<mint, mint>;
using vpmm = vt<pmm>;

typedef complex<double> cd;
template<class T>
struct Convolution { // given 2 array a and b, compute the number of pair that make up sum == x for every x from 1 to MX
    static void fft(vt<cd>& a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2 * M_PI / len * (invert ? -1 : 1);
            cd wlen(cos(angle), sin(angle));
            for (int i = 0; i < n; i += len) {
                cd w(1);
                for (int j = 0; j < len / 2; j++) {
                    cd u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            for (cd & x : a) x /= n;
        }
    }
 
    static vt<T> convolve(const vt<T>& a, const vt<T>& b) {
        vt<cd> fa(all(a)), fb(all(b));
        int n = 1;
        while (n < (int)a.size() + (int)b.size() - 1) n <<= 1;
        fa.rsz(n);
        fb.rsz(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++) fa[i] *= fb[i];
        fft(fa, true);
        vt<T> res(n);
        for (int i = 0; i < n; i++)
            res[i] = (T)round(fa[i].real());
        return res;
    }
};

template<int mod = MOD>
struct Polynomial {
    vmint a;
    Polynomial() {}
    Polynomial(const vmint& a) : a(a) { normalize(); }
    void normalize() {
        while (!a.empty() && a.back().value == 0) a.pop_back();
    }
    int size() const { return a.size(); }
    int deg() const { return a.size() - 1; }
    mint at(int i) const { return (i < 0 || i >= a.size()) ? mint(0) : a[i]; }
    static void fft(vmint& a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            mint wlen = mint(3).pow((mod - 1) / len);
            if (invert) wlen = wlen.inv();
            for (int i = 0; i < n; i += len) {
                mint w = 1;
                for (int j = 0; j < len / 2; j++) {
                    mint u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            mint inv_n = mint(n).inv();
            for (mint &x : a)
                x *= inv_n;
        }
    }
    static vmint multiply(const vmint& a, const vmint& b) {
        auto fa(a), fb(b);
        int n = 1;
        while (n < (int)a.size() + (int)b.size() - 1) n <<= 1;
        fa.rsz(n);
        fb.rsz(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++) fa[i] *= fb[i];
        fft(fa, true);
        vmint res(a.size() + b.size() - 1);
        for (int i = 0; i < res.size(); i++) res[i] = fa[i];
        return res;
    }
    Polynomial operator*(const Polynomial &other) const {
        vmint res = multiply(a, other.a);
        return Polynomial(res);
    }
};

struct BigCalculator {
    string num1;
    string num2;
    BigCalculator(const string n1, const string n2) : num1(n1), num2(n2) {}
    string add() { return bigAdd(num1, num2); }
    string minus() { return bigSub(num1, num2); }
    string multiply() { return bigMul(num1, num2); }
    string OR() { return bigBitOr(num1, num2); }
    string AND() { return bigBitAnd(num1, num2); }
    string XOR() { return bigBitXor(num1, num2); }
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
            result.push_back('0' + sum % 10);
        }
        reverse(result.begin(), result.end());
        return result;
    }
    static int compare(const string &a, const string &b) {
        if (a.size() != b.size())
            return (a.size() < b.size() ? -1 : 1);
        return a.compare(b);
    }
    static string bigSub(const string &a, const string &b) {
        if (compare(a, b) == 0)
            return "0";
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
            result.push_back('0' + diff);
            i--;
        }
        while (result.size() > 1 && result.back() == '0')
            result.pop_back();
        reverse(result.begin(), result.end());
        if (negative) result.insert(result.begin(), '-');
        return result;
    }
    typedef complex<double> cd;
    static void fft(vector<cd> &a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * M_PI / len * (invert ? -1 : 1);
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
            for (int i = 0; i < n; i++)
                a[i] /= n;
        }
    }
    static vector<int> multiplyFFT(const vector<int> &a, const vector<int> &b) {
        vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
        int n = 1;
        while (n < int(a.size() + b.size()))
            n <<= 1;
        fa.resize(n); fb.resize(n);
        fft(fa, false);
        fft(fb, false);
        for (int i = 0; i < n; i++)
            fa[i] *= fb[i];
        fft(fa, true);
        vector<int> result(n);
        int carry = 0;
        for (int i = 0; i < n; i++) {
            long long t = static_cast<long long>(round(fa[i].real())) + carry;
            result[i] = t % 10;
            carry = t / 10;
        }
        while (carry) {
            result.push_back(carry % 10);
            carry /= 10;
        }
        while (result.size() > 1 && result.back() == 0)
            result.pop_back();
        return result;
    }
    static string bigMul(const string &a, const string &b) {
        if (a == "0" || b == "0")
            return "0";
        vector<int> A(a.size()), B(b.size());
        for (size_t i = 0; i < a.size(); i++)
            A[a.size() - 1 - i] = a[i] - '0';
        for (size_t i = 0; i < b.size(); i++)
            B[b.size() - 1 - i] = b[i] - '0';
        vector<int> resultVec = multiplyFFT(A, B);
        string result;
        for (int i = resultVec.size() - 1; i >= 0; i--)
            result.push_back(resultVec[i] + '0');
        size_t pos = result.find_first_not_of('0');
        return (pos != string::npos) ? result.substr(pos) : "0";
    }
    static string decToBin(const string &dec) {
        if (dec == "0") return "0";
        string number = dec, bin;
        while (number != "0") {
            int rem;
            number = divideByTwo(number, rem);
            bin.push_back('0' + rem);
        }
        reverse(bin.begin(), bin.end());
        return bin;
    }
    static string binToDec(const string &bin) {
        string result = "0";
        for (char bit : bin) {
            result = bigMul(result, "2");
            if (bit == '1')
                result = bigAdd(result, "1");
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
            if (!quotient.empty() || digit != 0)
                quotient.push_back('0' + digit);
        }
        if (quotient.empty())
            quotient = "0";
        rem = carry;
        return quotient;
    }
    static string bigBitOr(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.push_back((binA[i] == '1' || binB[i] == '1') ? '1' : '0');
        return binRes;
    }
    static string bigBitAnd(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.push_back((binA[i] == '1' && binB[i] == '1') ? '1' : '0');
        return binRes;
    }
    static string bigBitXor(const string &a, const string &b) {
        size_t maxLen = max(a.size(), b.size());
        string binA = string(maxLen - a.size(), '0') + a;
        string binB = string(maxLen - b.size(), '0') + b;
        string binRes;
        for (size_t i = 0; i < maxLen; i++)
            binRes.push_back((binA[i] != binB[i]) ? '1' : '0');
        return binRes;
    }
};
