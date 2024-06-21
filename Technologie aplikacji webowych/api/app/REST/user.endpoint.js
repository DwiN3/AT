import business from '../business/business.container';
import applicationException from '../service/applicationException';
import auth from '../middleware/auth';

const userEndpoint = (router) => {
    router.post('/api/user/auth', async (request, response, next) => {
        try {
            let result = await business.getUserManager(request).authenticate(request.body.login, request.body.password);
            response.status(200).send(result);
        } catch (error) {
            applicationException.errorHandler(error, response);
        }
    });

    router.post('/api/user/create', async (request, response, next) => {
        try {
            const result = await business.getUserManager(request).createNewOrUpdate(request.body);
            response.status(200).send(result);
        } catch (error) {
            applicationException.errorHandler(error, response);
        }
    });

    router.delete('/api/user/logout/:userId', auth, async (request, response, next) => {
        try {
            let result = await business.getUserManager(request).removeHashSession(request.body.userId);
            response.status(200).send(result);
        } catch (error) {
            applicationException.errorHandler(error, response);
        }
    });

    router.post('/api/user/change-password/:userId', auth, async (request, response, next) => {
        try {
            const result = await business.getUserManager(request).changePassword(request.params.userId, request.body.newPassword);
            response.status(200).send(result);
        } catch (error) {
            applicationException.errorHandler(error, response);
        }
    });

    router.delete('/api/user/:userId', auth, async (request, response, next) => {
        try {
            const result = await business.getUserManager(request).removeUser(request.params.userId);
            response.status(200).send(result);
        } catch (error) {
            applicationException.errorHandler(error, response);
        }
    });
};

export default userEndpoint;