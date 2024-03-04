using System.Security.Cryptography;
using FeelFlowAnalysis.Models;

namespace FeelFlowAnalysis.Services;

/// <summary>
/// Service for hashing passwords and generating salts.
/// </summary> 
/// <remarks>
/// Initializes a new instance of the PasswordHasher class.
/// </remarks>
/// <param name="settings">Password hasher settings.</param>
public class Hashing(IHashingSettings settings) : IHashing
{
    private readonly int _saltSize = settings.SaltSize;
    private readonly int _hashSize = settings.HashSize;
    private readonly int _iterations = settings.Iterations;

    /// <summary>
    /// Hashes the input password using a salt.
    /// </summary>
    /// <param name="password">The password to be hashed.</param>
    /// <returns>The hashed password.</returns>
    public string HashString(string password)
    {
        byte[] salt = new byte[_saltSize];
        
        using (var rng = RandomNumberGenerator.Create())
            rng.GetBytes(salt);

        byte[] hash;

        // Using PBKDF2 with SHA256 for password hashing
        using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, _iterations, HashAlgorithmName.SHA256))
            hash = pbkdf2.GetBytes(_hashSize);

        // Combine salt and hash and convert to Base64 string
        return Convert.ToBase64String(salt) + ":" + Convert.ToBase64String(hash);
    }

    /// <summary>
    /// Verifies if the entered password matches the stored hashed password.
    /// </summary>
    /// <param name="storedPassword">The stored hashed password.</param>
    /// <param name="enteredPassword">The entered password for verification.</param>
    /// <returns>True if the passwords match, otherwise false.</returns>
    public bool VerifyString(string storedPassword, string enteredPassword)
    {
        // Split stored password into salt and hash parts
        var passwordParts = storedPassword.Split(':');
        var salt = Convert.FromBase64String(passwordParts[0]);
        var hash = Convert.FromBase64String(passwordParts[1]);

        // Using PBKDF2 with SHA256 for password verification
        using var pbkdf2 = new Rfc2898DeriveBytes(enteredPassword, salt, _iterations, HashAlgorithmName.SHA256);
        // Compare the derived hash with the stored hash
        return Enumerable.SequenceEqual(pbkdf2.GetBytes(_hashSize), hash);
    }
}
