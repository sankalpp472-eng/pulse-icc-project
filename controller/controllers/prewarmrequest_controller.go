package controllers

import (
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	knativeservingv1 "knative.dev/serving/pkg/apis/serving/v1"

	pulsev1alpha1 "pulse-icc-project/controller/api/v1alpha1"
)

type PrewarmRequestReconciler struct {
	client.Client
}

func (r *PrewarmRequestReconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var prewarm pulsev1alpha1.PrewarmRequest
	if err := r.Get(ctx, req.NamespacedName, &prewarm); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Validate the request.
	switch {
	case prewarm.Spec.Service == "":
		return r.setFailed(ctx, &prewarm, "spec.service must not be empty")
	case prewarm.Spec.TTLSeconds < 1:
		return r.setFailed(ctx, &prewarm, "spec.ttlSeconds must be at least 1")
	case prewarm.Spec.DesiredReplicas < 0 || prewarm.Spec.DesiredReplicas > 5:
		return r.setFailed(ctx, &prewarm, "spec.desiredReplicas must be between 0 and 5")
	case prewarm.Spec.Probability < 0 || prewarm.Spec.Probability > 1:
		return r.setFailed(ctx, &prewarm, "spec.probability must be between 0 and 1")
	case prewarm.Spec.Threshold < 0 || prewarm.Spec.Threshold > 1:
		return r.setFailed(ctx, &prewarm, "spec.threshold must be between 0 and 1")
	case prewarm.Spec.Probability < prewarm.Spec.Threshold:
		return r.setFailed(ctx, &prewarm, "probability is below the prewarming threshold")
	}

	// Look up the Knative Service in the same namespace.
	var service knativeservingv1.Service
	serviceKey := client.ObjectKey{
		Namespace: prewarm.Namespace,
		Name:      prewarm.Spec.Service,
	}

	if err := r.Get(ctx, serviceKey, &service); err != nil {
		if apierrors.IsNotFound(err) {
			return r.setFailed(
				ctx,
				&prewarm,
				fmt.Sprintf("Knative Service %q was not found", prewarm.Spec.Service),
			)
		}
		return ctrl.Result{}, err
	}

	now := time.Now()

	// Expiration is based on the later of requestedAt and lastActivityAt.
	baseTime := prewarm.Spec.RequestedAt.Time
	if prewarm.Status.LastActivityAt != nil &&
		prewarm.Status.LastActivityAt.Time.After(baseTime) {
		baseTime = prewarm.Status.LastActivityAt.Time
	}

	expiresAt := baseTime.Add(
		time.Duration(prewarm.Spec.TTLSeconds) * time.Second,
	)

	expiration := metav1.NewTime(expiresAt)
	prewarm.Status.ExpiresAt = &expiration

	if !now.Before(expiresAt) {
		prewarm.Status.Phase = "Evicted"
		prewarm.Status.Message = "Prewarming TTL expired; scale-down is not implemented yet"

		if err := r.Status().Update(ctx, &prewarm); err != nil {
			return ctrl.Result{}, err
		}

		logger.Info("Prewarm request expired", "service", prewarm.Spec.Service)
		return ctrl.Result{}, nil
	}

	// Scaling will be implemented in the next stage.
	prewarm.Status.Phase = "Pending"
	prewarm.Status.Message = "Knative Service found; scaling mechanism not implemented yet"

	if err := r.Status().Update(ctx, &prewarm); err != nil {
		return ctrl.Result{}, err
	}

	remaining := time.Until(expiresAt)
	if remaining < time.Second {
		remaining = time.Second
	}

	return ctrl.Result{RequeueAfter: remaining}, nil
}

func (r *PrewarmRequestReconciler) setFailed(
	ctx context.Context,
	prewarm *pulsev1alpha1.PrewarmRequest,
	message string,
) (ctrl.Result, error) {
	prewarm.Status.Phase = "Failed"
	prewarm.Status.Message = message

	if err := r.Status().Update(ctx, prewarm); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func (r *PrewarmRequestReconciler) SetupWithManager(
	manager ctrl.Manager,
) error {
	return ctrl.NewControllerManagedBy(manager).
		For(&pulsev1alpha1.PrewarmRequest{}).
		Complete(r)
}
