using MongoDB.Driver;
using FeelFlowAnalysis.Models.Entities;
using FeelFlowAnalysis.Services.Interfaces;
using Microsoft.Extensions.Options;

namespace FeelFlowAnalysis.Services.Implementations;

// Summary:
//     Service for managing user data.
public sealed class UserService(IOptions<Settings> settings, IMongoClient mongoClient) : IUserService
{   
    private readonly IMongoCollection<User> _users = mongoClient
        .GetDatabase(settings.Value.Database.Name)
        .GetCollection<User>(settings.Value.Database.UsersCollectionName);

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
        await (await _users.FindAsync(u => u.Email == email && u.PasswordHash == passwordHash))
            .FirstOrDefaultAsync();

    // Summary:
    //     Creates a new user.
    //
    // Parameters:
    //   user:
    //     The user to create.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    public async Task Create(User user) => await _users.InsertOneAsync(user);

    // Summary:
    //     Gets all users.
    //
    // Returns:
    //     A task that represents the asynchronous operation. The task result contains
    //     a list of all users.
    public async Task<List<User>> GetAll() => await (await _users.FindAsync(u => true)).ToListAsync();

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
        await (await _users.FindAsync(user => user.Id == id)).FirstOrDefaultAsync();

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
        await (await _users.FindAsync(user => user.Email == email)).FirstOrDefaultAsync();

    // Summary:
    //     Removes a user by ID.
    //
    // Parameters:
    //   id:
    //     The ID of the user to remove.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    public async Task Remove(string id) => await _users.DeleteOneAsync(u => u.Id == id);

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
    public async Task Update(string id, User user) => await _users.ReplaceOneAsync(u => u.Id == id, user);
}
