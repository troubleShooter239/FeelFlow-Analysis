using System.Security.Cryptography;
using FeelFlowAnalysis.Models;

namespace FeelFlowAnalysis.Services;

/// <summary>
/// Implementation of IAesEncryptor using AES encryption algorithm.
/// </summary>
public class Encryption : IEncryption
{
    private readonly Aes _aes = Aes.Create();

    /// <summary>
    /// Initialize the AES algorithm with pre-defined key and IV.
    /// </summary>
    /// <param name="settings">Aes encryptor settings.</param>
    public Encryption(IEncryptionSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        if (string.IsNullOrEmpty(settings.EncryptionKey) || settings.EncryptionKey.Length != 32)
            throw new ArgumentException(
                "Encryption key must be a Base64-encoded 32-byte string.", nameof(settings.EncryptionKey)
            );

        if (string.IsNullOrEmpty(settings.InitializationVector) || settings.InitializationVector.Length != 16)
            throw new ArgumentException(
                "Initialization vector must be a Base64-encoded 16-byte string.", nameof(settings.InitializationVector)
            );

        _aes.Key = Convert.FromBase64String(settings.EncryptionKey);
        _aes.IV = Convert.FromBase64String(settings.InitializationVector);
    }

    /// <summary>
    /// Encrypt the input string using AES algorithm and return the Base64-encoded ciphertext.
    /// </summary>
    /// <param name="value">The value to be encrypted.</param>
    /// <returns>The encrypted data.</returns>
    public string EncryptString(string value)
    {
        using var encryptor = _aes.CreateEncryptor();
        using var msEncrypt = new MemoryStream();
        using (var csEncrypt = new CryptoStream(msEncrypt, encryptor, CryptoStreamMode.Write))
        using (var swEncrypt = new StreamWriter(csEncrypt, System.Text.Encoding.UTF8))
        {
            swEncrypt.Write(value);
        }

        return Convert.ToBase64String(msEncrypt.ToArray());
    }

    /// <summary>
    /// Verify if the stored encrypted data matches the encrypted form of the entered data.
    /// </summary>
    /// <param name="storedData">The encrypted stored data.</param>
    /// <param name="enteredData">The entered data for verification.</param>
    /// <returns>True if the data match, otherwise false.</returns>
    public bool VerifyString(string storedData, string enteredData)
        => storedData == EncryptString(enteredData);
}