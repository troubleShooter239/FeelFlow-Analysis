using MongoDB.Driver;
using FeelFlowAnalysis.Models.Entities;
using FeelFlowAnalysis.Models.Settings;
using FeelFlowAnalysis.Services.Interfaces;


namespace FeelFlowAnalysis.Services.Implementations;

// Summary:
//     Service for managing user data.
public class UserService(IDbSettings settings, IMongoClient mongoClient) : IUserService
{   
    private readonly IMongoCollection<User> _users = mongoClient
        .GetDatabase(settings.DatabaseName)
        .GetCollection<User>(settings.UsersCollectionName);

    // Summary:
    //     Authenticates a user based on email and password hash.
    //
    // Parameters:
    //   email:
    //     The email of the user.
    //
    //   passwordHash:
    //     The hashed password of the user.
    //
    // Returns:
    //     A task that represents the asynchronous authentication operation. The task
    //     result contains the authenticated user, if authentication succeeds; otherwise,
    //     null.
    public async Task<User> Authenticate(string email, string passwordHash) => 
        (await _users.FindAsync(u => u.Email == email && u.PasswordHash == passwordHash)).FirstOrDefault();

    // Summary:
    //     Creates a new user.
    //
    // Parameters:
    //   user:
    //     The user to create.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    public async Task Create(User user) =>
        await _users.InsertOneAsync(user);

    // Summary:
    //     Gets all users.
    //
    // Returns:
    //     A task that represents the asynchronous operation. The task result contains
    //     a list of all users.
    public async Task<List<User>> GetAll() => 
        (await _users.FindAsync(u => true)).ToList();

    // Summary:
    //     Gets a user by ID.
    //
    // Parameters:
    //   id:
    //     The ID of the user to retrieve.
    //
    // Returns:
    //     A task that represents the asynchronous operation. The task result contains
    //     the user with the specified ID.
    public async Task<User> Get(string id) =>
        (await _users.FindAsync(user => user.Id == id)).FirstOrDefault();

    // Summary:
    //     Gets a user by email.
    //
    // Parameters:
    //   email:
    //     The email of the user to retrieve.
    //
    // Returns:
    //     A task that represents the asynchronous operation. The task result contains
    //     the user with the specified email.
    public async Task<User> GetByEmail(string email) => 
        (await _users.FindAsync(user => user.Email == email)).FirstOrDefault();

    // Summary:
    //     Removes a user by ID.
    //
    // Parameters:
    //   id:
    //     The ID of the user to remove.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    public async Task Remove(string id) => 
        await _users.DeleteOneAsync(u => u.Id == id);

    // Summary:
    //     Updates an existing user.
    //
    // Parameters:
    //   id:
    //     The ID of the user to update.
    //
    //   user:
    //     The updated user object.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    public async Task Update(string id, User user) => 
        await _users.ReplaceOneAsync(u => u.Id == id, user);
}
