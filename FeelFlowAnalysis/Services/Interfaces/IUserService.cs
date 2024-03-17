using FeelFlowAnalysis.Models.Entities;

namespace FeelFlowAnalysis.Services.Interfaces;

// Summary:
//     Interface for user service.
public interface IUserService
{
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
    Task<User> Authenticate(string email, string passwordHash);

    // Summary:
    //     Gets all users.
    //
    // Returns:
    //     A task that represents the asynchronous operation. The task result contains
    //     a list of all users.
    Task<List<User>> GetAll();

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
    Task<User> Get(string id);

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
    Task<User> GetByEmail(string email);

    // Summary:
    //     Creates a new user.
    //
    // Parameters:
    //   user:
    //     The user to create.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    Task Create(User user);

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
    Task Update(string id, User user);

    // Summary:
    //     Removes a user by ID.
    //
    // Parameters:
    //   id:
    //     The ID of the user to remove.
    //
    // Returns:
    //     A task that represents the asynchronous operation.
    Task Remove(string id);
}
