using MongoDB.Driver;
using FeelFlowAnalysis.Models;
namespace FeelFlowAnalysis.Services;

/// <summary>
/// Initializes a new instance of the UserService class.
/// </summary>
/// <param name="settings">The database settings.</param>
/// <param name="mongoClient">The MongoDB client.</param>
public class UserService(IDbSettings settings, IMongoClient mongoClient) : IUserService
{   
    private readonly IMongoCollection<User> _users = mongoClient
            .GetDatabase(settings.DatabaseName)
            .GetCollection<User>(settings.UsersCollectionName);

    /// <summary>
    /// Authenticates a user based on email and password.
    /// </summary>
    /// <param name="email">The email address of the user.</param>
    /// <param name="passwordHash">The hashed password of the user.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the user.</returns>
    public async Task<User> Authenticate(string email, string passwordHash)
        => (await _users.FindAsync(u => u.Email == email && u.PasswordHash == passwordHash)).FirstOrDefault();
    
    /// <summary>
    /// Creates a new user.
    /// </summary>
    /// <param name="user">The user to create.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public async Task Create(User user) 
        => await _users.InsertOneAsync(user);

    /// <summary>
    /// Retrieves all users.
    /// </summary>
    /// <returns>A task that represents the asynchronous operation. The task result contains the list of users.</returns>
    public async Task<List<User>> GetAll()
        => (await _users.FindAsync(u => true)).ToList();

    /// <summary>
    /// Retrieves a user by ID.
    /// </summary>
    /// <param name="id">The ID of the user to retrieve.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the user.</returns>
    public async Task<User> Get(string id)
        => (await _users.FindAsync(user => user.Id == id)).FirstOrDefault();

    /// <summary>
    /// Retrieves a user by email address.
    /// </summary>
    /// <param name="email">The email address of the user to retrieve.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the user.</returns>
    public async Task<User> GetByEmail(string email)
        => (await _users.FindAsync(user => user.Email == email)).FirstOrDefault();

    /// <summary>
    /// Removes a user by ID.
    /// </summary>
    /// <param name="id">The ID of the user to remove.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public async Task Remove(string id)
        => await _users.DeleteOneAsync(u => u.Id == id);

    /// <summary>
    /// Updates a user by ID.
    /// </summary>
    /// <param name="id">The ID of the user to update.</param>
    /// <param name="user">The updated user data.</param>
    /// <returns>A task that represents the asynchronous operation.</returns>
    public async Task Update(string id, User user)
        => await _users.ReplaceOneAsync(u => u.Id == id, user);
}
